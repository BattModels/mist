#!/usr/bin/env -S julia --project --threads=1 --startup-file=no
using TokenizerStats: main
exit(main(ARGS))
