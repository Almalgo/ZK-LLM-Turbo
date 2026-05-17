import React from "react";
import Button from "@mui/material/Button";
import Grid from "@mui/material/Grid";

import OutlinedTextArea from "../../common/OutlinedTextArea";
import "./style.css";

const details = [
  "Session(public_context_b64) -> session_id",
  "Layer(session_id, layer_idx, operation, encrypted_vectors_b64) -> encrypted_results_b64",
  "",
  "Supported layer operations:",
  "- qkv",
  "- o_proj",
  "- ffn_gate_up",
  "- ffn_down",
  "- ffn_merged",
].join("\n");

export default class ZKLLMTurboDemo extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      showDetails: false,
    };
  }

  toggleDetails = () => {
    this.setState((state) => ({ showDetails: !state.showDetails }));
  };

  renderDetails() {
    if (!this.state.showDetails) {
      return null;
    }

    return (
      <div className="zkPanel">
        <h4>Published API</h4>
        <OutlinedTextArea value={details} rows={8} fullWidth={true} />
      </div>
    );
  }

  render() {
    return (
      <Grid container direction="column" spacing={2} className="zkDemo">
        <Grid item>
          <div className="zkHeader">
            <h3>ZK-LLM Turbo</h3>
            <p>Privacy-preserving split inference with CKKS encrypted layer calls.</p>
          </div>
        </Grid>

        <Grid item>
          <div className="zkPanel">
            <h4>What this service does</h4>
            <p>
              ZK-LLM Turbo keeps sensitive embedding vectors encrypted while the hosted
              service performs supported linear layer operations on CKKS ciphertexts.
            </p>
            <Button variant="contained" color="primary" onClick={this.toggleDetails}>
              {this.state.showDetails ? "Hide request shape" : "Show request shape"}
            </Button>
          </div>
        </Grid>

        <Grid item>{this.renderDetails()}</Grid>

        <Grid item>
          <div className="zkPanel zkMuted">
            <h4>Browser demo scope</h4>
            <p>
              Full encrypted inference requires CKKS context creation and encrypted vector
              construction. Those steps are supported by the Python client and intentionally
              kept out of this browser UI.
            </p>
          </div>
        </Grid>
      </Grid>
    );
  }
}
