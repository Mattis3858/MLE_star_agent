"""Back-compat entry point. The agent now lives in the mle_star/ package.

    python -m mle_star_agent      (this shim)
    python -m mle_star            (canonical; supports --resume, --max-iter)
"""

import sys

from mle_star.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
