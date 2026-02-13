"""
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  https://www.apache.org/licenses/LICENSE-2.0
 *
 * See the NOTICE file distributed with this work for additional information
 * regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

import datetime
import gzip
import logging
import multiprocessing as mp
import psutil
import os

import calculon
from calculon.util import pick, arg_true_false_all
from calculon.llm import *


class OptimalExecution_MoE(calculon.CommandLine):
  NAME = 'llm-optimal-execution-moe'
  ALIASES = ['loe-moe']

  @staticmethod
  def create_parser(subparser):
    sp = subparser.add_parser(
      OptimalExecution_MoE.NAME, aliases=OptimalExecution_MoE.ALIASES,
      help='run a search to find the optimal llm execution')
    sp.set_defaults(func=OptimalExecution_MoE.run_command)
    sp.add_argument('-d', '--debug', action='store_true',
                    help='Loop over executions, don\'t run them')
    sp.add_argument('application', type=str,
                    help='File path to application configuration')
    sp.add_argument('num_procs', type=int,
                    help='Number of processors in execution')
    sp.add_argument('max_batch_size', type=int,
                    help='Maximum batch size, will be largest multiple of DP')
    sp.add_argument('datatype', type=str, choices=System.supported_datatypes(),
                    help='The datatype to use')
    sp.add_argument('system', type=str,
                    help='File path to system configuration')
    sp.add_argument('output', type=str,
                    help='File path to the output file'
                    " ('*.csv', '*.csv.gz', '*.json', '*.json.gz')")
    sp.add_argument('-c', '--cpus', type=int, default=psutil.cpu_count(logical=False),
                    help='CPUs to use for parallelization')
    sp.add_argument('-n', '--noneok', action='store_true',
                    help='Don\'t give failure status when no good execution exists')
    sp.add_argument('-t', '--top-n', type=int, default=1,
                    help='Number of best outputs')
    sp.add_argument('-l', '--layers', action='store_true',
                    help='Include layers information in output stats file')
    sp.add_argument('-f', '--fused_activation', type=arg_true_false_all,
                    default='true', help='Mode of fused activation')
    sp.add_argument('--no-tp-overlap', action='store_true',
                    help='Don\'t allow TP overlap')
    sp.add_argument('--no-dp-overlap', action='store_true',
                    help='Don\'t allow DP overlap')
    sp.add_argument('-moe', '--moe', type=int, default=16,
                    help='Number of experts')

  @staticmethod
  def run_command(logger, args):
    assert args.top_n > 0, 'top-n must be > 0'

    app = Llm.Application(calculon.io.read_json_file(args.application))
    syst = System(calculon.io.read_json_file(args.system))

    # Change the parameter of number of experts
    num_experts = args.moe
    
    # microbatch_size = 32
    params = []
      
    batch_size = 1024
    # dp =  batch_size // microbatch_size
    
    print('Running Configs')
    #print(args.num_procs)

    count = 0

    for ep in Llm.get_all_expert_parallelisms(num_experts, args.num_procs):
      for tp in Llm.get_all_tensor_parallelisms(
        args.num_procs // ep, app.hidden, app.attn_heads):
        # Hidden size must be multiples of tp
        es = tp
        if (not (app.hidden % tp == 0)):
          continue
        #print(tp)
        for pp in Llm.get_all_pipeline_parallelisms(args.num_procs, tp*ep, app.num_blocks):
          if( app.num_blocks % pp != 0 or args.num_procs%(ep*tp*pp)!=0) :
            continue
          # Get the number of processors left
          procs_left_nonMoE = args.num_procs // pp // tp
          procs_left_MoE = args.num_procs // tp // ep // pp
          # Change the batch size and get script for different conditions
          # We can make a graph 
          batch_size = OptimalExecution.get_batch_size(procs_left_nonMoE, args.max_batch_size)
          if (batch_size == None):
            continue
          if(procs_left_nonMoE > batch_size):
              #print("Process left ", procs_left_nonMoE, "batch size ", batch_size)
              procs_left_nonMoE = batch_size
          if((not (batch_size % procs_left_nonMoE == 0))):
              continue
          dp = procs_left_nonMoE
          dp_exp = procs_left_MoE
          #print(dp)
          # microbatch_size = batch_size / procs_left
          for ppint in Llm.get_valid_pipeline_interleavings(app.num_blocks, pp):
            #print(f'pp: {pp}, ep: {ep}, tp: {tp}, dp: {dp}, dp_exp: {dp_exp}, ppint: {ppint}')
            # attn_only is not supported for now.
            for activation_recompute in ['none', 'full', 'attn_only']:
            #for activation_recompute in ['none']:
                for optimizer_sharding in pick(dp>1, [True, False], [False]):
                    for tensor_par_comm_type in ['ar', 'p2p_rs_ag', 'rs_ag']:
                        count += 1
                        params.append(
                        (args, args.debug, args.top_n, args.layers, args.num_procs,
                        args.datatype, num_experts, app, syst, ep, es, tp, pp, dp, dp_exp,
                        ppint, batch_size, activation_recompute, optimizer_sharding,
                        tensor_par_comm_type, args.fused_activation,
                        not args.no_tp_overlap, not args.no_dp_overlap))

    print(f'number of param combinations: {len(params)}')
    # Runs parallel searches
    start_time = datetime.datetime.now()
    with mp.Pool(args.cpus) as pool:
      searches = pool.starmap(OptimalExecution_MoE.search, params, chunksize=20)
    end_time = datetime.datetime.now()
    
    # Combines parallel search result into one data structure
    best = []
    exe_count = 0
    good_exe_count = 0
    bad_exe_count = 0
    for cbest, ec, gec, bec, tp, pp in searches:
      best = OptimalExecution_MoE.update_list(best, cbest, args.top_n)
      exe_count += ec
      good_exe_count += gec
      bad_exe_count += bec

    logger.info(f'Total executions: {exe_count}')
    logger.info(f'Good executions: {good_exe_count}')
    logger.info(f'Bad executions: {bad_exe_count}')
    calc_rate = exe_count / (end_time - start_time).total_seconds()
    logger.info(f'Calculation rate: {calc_rate:.2f} calcs/sec')
    if args.debug:
      return 0

    if len(best) == 0:
      if not args.noneok:
        logger.fatal('No acceptable configurations found :(')
        return -1
      else:
        logger.info('No acceptable configurations found :(')
    else:
      logger.info(f'Best sample rate: {best[0][0]}')

    output = {}
    for index, run in enumerate(best):
      _, execution, stats = run
      output[index] = {
        'execution': execution,
        'stats': stats
      }

    if calculon.io.is_json_extension(args.output):
      logger.info(f'Output: {args.output}')
      calculon.io.write_json_file(output, args.output)
    elif args.output.endswith('.csv') or args.output.endswith('.csv.gz'):
      logger.info(f'Output: {args.output}')
      exe_keys = list(output[0]['execution'].keys())
      stats_keys = list(output[0]['stats'].keys())
      opener = gzip.open if args.output.endswith('.gz') else open
      with opener(args.output, 'wb') as fd:
        fd.write(bytes(f',{",".join(exe_keys)},{",".join(stats_keys)}\n',
                       'utf-8'))
        for index in sorted(output.keys()):
          fd.write(bytes(f'{index}', 'utf-8'))
          for exe_key in exe_keys:
            fd.write(bytes(f',{output[index]["execution"][exe_key]}', 'utf-8'))
          for stats_key in stats_keys:
            fd.write(bytes(f',{output[index]["stats"][stats_key]}', 'utf-8'))
          fd.write(bytes('\n', 'utf-8'))
    else:
      assert False, f'Unknown file type: {args.output}'

    return 0

  # @staticmethod
  # def get_batch_size(data_par, max_batch_size):
  #   if data_par > max_batch_size:
  #     return None
  #   last = data_par
  #   while True:
  #     if last + data_par > max_batch_size:
  #       return last
  #     else:
  #       last += data_par

  @staticmethod
  def search(args, debug, top_n, layers, num_procs, datatype, num_experts,
             app, syst, ep, es, tp, pp, dp, dp_exp, ppint, batch_size, activation_recompute,
             optimizer_sharding, tensor_par_comm_type, fused_acts,
             allow_tp_overlap, allow_dp_overlap):
    num_nets = syst.num_networks

    best = []
    exe_count = 0
    good_exe_count = 0
    bad_exe_count = 0

    has_mem2 = syst.mem2.capacity > 0

    can_redo = Llm.can_redo_ag(tensor_par_comm_type,
                               activation_recompute)
    use_MOE = True
    if num_experts == 1:
        use_MOE = False
    top_k_experts = min(2, num_experts)
    # Reduce the search space to get the best runtime and parallelization only
    for seq_par_ag_redo in pick(False, [True, False], [False]):
      for data_par_overlap in pick(dp>1 and allow_dp_overlap, [True, False],
                                   [False]):
        for tensor_par_overlap in pick(tp>1 and allow_tp_overlap,
                                       ['none', 'ring', 'pipe'], ['none']):
          for weight_offload in pick(False, [True, False], [False]):
            if activation_recompute == 'full' or not has_mem2:
              activations_offloads = [False]
            else:
              activations_offloads = [True, False]
            for activations_offload in [False]:
              for optimizer_offload in pick(False, [True, False],
                                            [False]):
                for fused_act in fused_acts:
                  for microbatch_size in Llm.get_valid_microbatch_sizes(
                  app.seq_size, tp, dp, batch_size, pp):
                    for tn in pick(tp>1, range(num_nets), [0]):
                      for pn in pick(pp>1, range(num_nets), [0]):
                        for dn in pick(dp>1, range(num_nets), [0]):
                          for dn_exp in pick(dp_exp>1, range(num_nets), [0]):
                            for epn in pick(ep>1, range(num_nets), [0]):
                              for esn in pick(es>1, range(num_nets), [0]):
                                exe_count += 1
                                exe_json = {
                                  'num_procs': num_procs,
                                  'tensor_par': tp,
                                  'pipeline_par': pp,
                                  'data_par': dp,
                                  'tensor_par_net': tn,
                                  'pipeline_par_net': pn,
                                  'data_par_net': dn,
                                  'batch_size': batch_size,
                                  'microbatch_size': microbatch_size,
                                  "num_experts": num_experts,
                                  "top_k_experts": top_k_experts,
                                  "expert_par": ep,
                                  "expert_slice": es,
                                  "data_par_exp": dp_exp,
                                  "data_par_exp_net": dn_exp,
                                  "expert_par_net": epn,
                                  "expert_slice_net": esn,
                                  'datatype': datatype,
                                  'fused_activation': fused_act,
                                  'attention_type': 'multihead',
                                  'activation_recompute': activation_recompute,
                                  'pipeline_interleaving': ppint,
                                  'optimizer_sharding': optimizer_sharding,
                                  'tensor_par_comm_type': tensor_par_comm_type,
                                  'tensor_par_overlap': tensor_par_overlap,
                                  'seq_par_ag_redo': seq_par_ag_redo,
                                  'data_par_overlap': data_par_overlap,
                                  'weight_offload': weight_offload,
                                  'activations_offload': activations_offload,
                                  'optimizer_offload': optimizer_offload,
                                  'training': True,
                                  'model_MoE': True
                                }
                                
                                if not debug:
                                  try:
                                    logger = logging.Logger('sub')
                                    model = Llm(app, logger)
                                    model.compile(
                                      syst,
                                      Llm.Execution.from_json(exe_json))
                                    model.run(syst)
                                    stats = model.get_stats_json(layers)
                                    good_exe_count += 1
                                    curr = (stats['sample_rate'], exe_json, stats)
                                    best = OptimalExecution_MoE.update_list(best, curr,
                                                                        top_n)
                                    # if(ep == 8 and ep == 8 and es == 1 and tp == 8 and dp == 512 and ppint == 5):
                                    # print(f'best sample rate is : {best[0][0]}')
                                    #if(ep == 8 and ep == 8 and es == 1 and tp == 8 and dp == 512 and ppint == 5):
                                    #  print(f'success: {curr[0]}')
                                  except Llm.Error as ex:
                                    logger = logging.getLogger()
                                    logger.debug(f'JSON:{exe_json}\nERROR:{ex}\n')
                                    bad_exe_count += 1
                    # if mbs_break and good_exe_count == mbs_break_good:
                    #   break
    return (best, exe_count, good_exe_count, bad_exe_count, tp, pp)

  @staticmethod
  def update_list(current, candidate, quantity):
    if not isinstance(candidate, list):
      current.append(candidate)
    else:
      current.extend(candidate)
    current.sort(reverse=True, key=lambda x: x[0])
    return current[:quantity]


calculon.CommandLine.register(OptimalExecution_MoE)
