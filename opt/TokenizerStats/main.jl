#!/usr/bin/env -S julia --project --threads=4 --startup-file=no
using TokenizerStats: main
exit(main(ARGS))
