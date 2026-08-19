# Data availability

This project does not redistribute ADNI participant-level data. Researchers must obtain approval
from ADNI and place their authorized files below a local directory referenced by
`STORM_DATA_ROOT`.

Only fully synthetic fixtures may be committed under `data/demo/`. Real identifiers such as PTID,
RID, LONIUID, scan dates, clinical fields, ROI vectors, and participant-aligned derived outputs
must remain outside version control.

