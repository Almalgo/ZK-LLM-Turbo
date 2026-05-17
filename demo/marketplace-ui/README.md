# ZK-LLM Turbo Marketplace Demo

Upload `zk-llm-turbo-marketplace-demo.zip` in Publisher as the service demo UI archive.

This demo uses the `Health` RPC from `snet_service/proto/zk_llm_http_api.proto`.
It intentionally avoids browser-side CKKS key generation and encrypted vector
construction so Marketplace users can verify that the live hosted service is up
with one click.

Files:

- `index.js`: UI Sandbox React component.
- `style.css`: Demo styling.
- `zk_llm_http_api_pb.js`: Generated protobuf messages.
- `zk_llm_http_api_pb_service.js`: Generated service descriptors.

Publisher notes:

1. Upload the updated proto file: `snet_service/proto/zk_llm_http_api.proto`.
2. Upload the demo zip file.
3. Wait a few minutes for Publisher to process the UI archive.
