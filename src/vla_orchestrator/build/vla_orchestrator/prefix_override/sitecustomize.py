import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/psh/Robot1/src/vla_orchestrator/install/vla_orchestrator'
