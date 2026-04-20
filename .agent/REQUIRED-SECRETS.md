# Required Runtime Values for Milestone 5

## Values You Must Provide

- `ORG_ID`
- `SERVICE_ID`
- `DOMAIN`
- `SEPOLIA_RPC_URL`
- `MAINNET_RPC_URL`
- Signer/wallet details (Sepolia + Mainnet)
- Gas/funding confirmation (Sepolia + Mainnet)

## Where to Get Them

- `ORG_ID`, `SERVICE_ID`:
  - SingularityNet Publisher Portal (organization/service pages), or
  - `snet` CLI service/org listing commands
- `DOMAIN`:
  - your DNS provider (Cloudflare/Route53/etc.) pointing to daemon host
- RPC URLs:
  - Infura, Alchemy, QuickNode, etc.
  - one endpoint for Sepolia, one for Mainnet

## Security Handling

- Do not commit filled config files containing sensitive values.
- Use template copy workflow in `snet_service/README.md`.
- Local filled configs are gitignored:
  - `snet_service/snetd.config.sepolia.json`
  - `snet_service/snetd.config.mainnet.json`
  - `snet_service/snetd.config.local.json`
