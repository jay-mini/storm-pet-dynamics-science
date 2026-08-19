# Model release policy

Model artifacts are private by default. A release requires all of the following:

1. Provenance, configuration, input schema, and SHA-256 checksums are complete.
2. The model loads and reproduces the registered regression fixture in a clean environment.
3. Participant-level predictions and training tables are absent from the deploy bundle.
4. Data-governance review confirms that the artifact may be distributed under the applicable
   agreement.
5. A model card records intended research use, limitations, OOD behavior, and revocation details.

Native pySuStaIn pickle files remain in the private run archive. Deployment should prefer JSON and
NumPy archives loaded with `allow_pickle=False`. A legacy pickle may only be loaded from a trusted,
read-only model root after checksum verification; it must never be accepted from user input.

