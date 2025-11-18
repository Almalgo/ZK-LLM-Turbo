from client.encryption.ckks_context import create_ckks_context
import tenseal as ts

def test_ckks_context_creation():
    ctx = create_ckks_context()
    assert isinstance(ctx, ts.Context)
    assert ctx.global_scale > 0
