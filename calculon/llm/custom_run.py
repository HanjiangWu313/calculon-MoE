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

import calculon
from calculon.llm import *
import matplotlib.pyplot as plt

class CustomRunner(calculon.CommandLine):
  NAME = 'llm-custom-run'
  ALIASES = ['lcr']

  @staticmethod
  def create_parser(subparser):
    sp = subparser.add_parser(CustomRunner.NAME, aliases=CustomRunner.ALIASES,
                              help='run a custom llm calculation')
    # The set defaults
    sp.set_defaults(func=CustomRunner.run_command)
    sp.add_argument('application', type=str,
                    help='File path to application configuration')
    sp.add_argument('execution', type=str,
                    help='File path to execution configuration')
    sp.add_argument('system', type=str,
                    help='File path to system configuration')
    sp.add_argument('stats', type=str,
                    help='File path to stats output ("-" for stdout")')
    sp.add_argument('-p', '--peers', type=str, default=None,
                    help='File path to write out peers file')
    sp.add_argument('-l', '--layers', action='store_true',
                    help='Include layers information in output stats file')

  @staticmethod
  def run_command(logger, args):
    app_json = calculon.io.read_json_file(args.application)
    exe_json = calculon.io.read_json_file(args.execution)
    sys_json = calculon.io.read_json_file(args.system)

    app = Llm.Application(app_json)
    exe = Llm.Execution.from_json(exe_json)
    syst = System(sys_json)
    
    num_procs = 64
    
    config_list = []
    batch_time_list = []
    compute_eff_list = []
    for ep in range(1, 17): 
      if ep > num_procs:
          break
      # if num_procs % ep != 0:
      #   continue
      for dp in range(1, 9):
        if dp > num_procs // ep:
          continue
        # if num_procs % (dp * ep) != 0:
        #   continue
        for pp in range(1, 33):
          if dp > num_procs // (ep * pp):
            continue
          # if num_procs % (pp * dp * ep) != 0:
          #   continue
          for tp in range(1, num_procs+1):
            if tp > num_procs // (dp * pp * ep):
              continue
            # if num_procs % (pp * dp * ep * tp) != 0:
            #   continue
            if(app.attn_heads % tp != 0):
              continue
            if (dp * pp * tp * ep == num_procs):
              # Inside the inner-most loop, we run the llm
              # We need to modify the corresponding 
              exe.tensor_par = tp
              exe.data_par = dp
              exe.pipeline_par = pp
              # exe.num_procs = ()
              print(f'par config - ep: {ep}, dp: {dp}, pp: {pp}, tp: {tp}')
              try:
                model = Llm(app, logger)
                model.compile(syst, exe)
                model.run(syst)
              except Llm.Error as error:
                # Print oyt the error
                print(f'ERROR: {error}')
              if args.stats == '-':
                model.display_throughput()
              config_list.append((f'ep: {ep}, dp: {dp}, pp: {pp}, tp: {tp}'))
              batch_time_list.append(model.get_total_time())
              compute_eff_list.append(model.get_total_efficiency()*100)
            tp *= 2  # Double the tp
          pp *= 2  # Double the pp
        dp *= 2  # Double the dp
      ep*= 2 # Double the ep

    # if args.stats == '-':
    #   model.display_stats()
    # elif calculon.is_json_extension(args.stats):
    #   calculon.write_json_file(model.get_stats_json(args.layers), args.stats)
    # else:
    #   assert False, f'unknown stats extension: {args.stats}'

    
    # if args.peers:
    #   calculon.write_json_file(exe.get_peers_json(), args.peers)
    
    plt.scatter(batch_time_list, compute_eff_list)
    for i, config in enumerate(config_list):
      plt.annotate(f'{config}', (batch_time_list[i], compute_eff_list[i]), textcoords="offset points", xytext=(0,10), ha='center',fontsize=5)
    plt.xlabel('batch_time')
    plt.ylabel('compute_efficiency')
    plt.title('GPT1.8T Inference Calculon')
    plt.savefig('gpt1.8T.png', dpi=300, bbox_inches='tight')
    return 0


calculon.CommandLine.register(CustomRunner)
