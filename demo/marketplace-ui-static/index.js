import React, { useState } from "react";
import StyledButton from "@integratedComponents/StyledButton";
import OutlinedTextArea from "@commonComponents/OutlinedTextArea";
import "./style.css";

const Demo = () => {
  const [showDetails, setShowDetails] = useState(false);

  return (
    <div className="zk-demo">
      <div className="zk-header">
        <h3>ZK-LLM Turbo</h3>
        <p>Privacy-preserving split inference with CKKS encrypted layer calls.</p>
      </div>

      <div className="zk-panel">
        <h4>What this service does</h4>
        <p>
          ZK-LLM Turbo keeps sensitive embedding vectors encrypted while the
          hosted service performs supported linear layer operations on CKKS
          ciphertexts.
        </p>
        <StyledButton
          btnText={showDetails ? "Hide details" : "Show request shape"}
          variant="contained"
          onClick={() => setShowDetails(!showDetails)}
        />
      </div>

      {showDetails ? (
        <div className="zk-panel">
          <h4>Published API</h4>
          <OutlinedTextArea
            value={
              "Session(public_context_b64) -> session_id\n" +
              "Layer(session_id, layer_idx, operation, encrypted_vectors_b64) -> encrypted_results_b64\n\n" +
              "Supported layer operations:\n" +
              "- qkv\n" +
              "- o_proj\n" +
              "- ffn_gate_up\n" +
              "- ffn_down\n" +
              "- ffn_merged"
            }
          />
        </div>
      ) : null}

      <div className="zk-panel zk-muted">
        <h4>Browser demo scope</h4>
        <p>
          Full encrypted inference requires CKKS context creation and encrypted
          vector construction. Those steps are supported by the Python client and
          are intentionally kept out of the Marketplace browser UI.
        </p>
      </div>
    </div>
  );
};

export default Demo;
