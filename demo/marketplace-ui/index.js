import React, { useState } from "react";
import StyledButton from "@integratedComponents/StyledButton";
import OutlinedTextArea from "@commonComponents/OutlinedTextArea";
import { ZKLLMService } from "./zk_llm_http_api_pb_service";
import "./style.css";

const Demo = ({ serviceClient, isComplete }) => {
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isRunning, setIsRunning] = useState(false);

  const callHealth = () => {
    setError("");
    setResult(null);
    setIsRunning(true);

    const methodDescriptor = ZKLLMService.Health;
    const request = new methodDescriptor.requestType();
    request.setOp("health");

    serviceClient.unary(methodDescriptor, {
      request,
      preventCloseServiceOnEnd: false,
      onEnd: (response) => {
        setIsRunning(false);
        const { message, status, statusMessage } = response;

        if (status !== 0) {
          setError(statusMessage || "Service call failed");
          return;
        }

        setResult({
          status: message.getStatus(),
          service: message.getService(),
          model: message.getModel(),
          modelStatus: message.getModelStatus(),
          modelError: message.getModelError(),
        });
      },
    });
  };

  const ResultView = () => {
    if (error) {
      return (
        <div className="zk-panel zk-error">
          <h4>Service call failed</h4>
          <OutlinedTextArea value={error} />
        </div>
      );
    }

    if (!result) {
      return (
        <div className="zk-panel zk-muted">
          <h4>Output</h4>
          <p>Run the demo to check the live hosted service status.</p>
        </div>
      );
    }

    return (
      <div className="zk-panel">
        <h4>Live service response</h4>
        <div className="zk-result-grid">
          <span>Status</span>
          <strong>{result.status || "unknown"}</strong>
          <span>Service</span>
          <strong>{result.service || "zk-llm-turbo"}</strong>
          <span>Model</span>
          <strong>{result.model || "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}</strong>
          <span>Model readiness</span>
          <strong>{result.modelStatus || "not_loaded"}</strong>
        </div>
        {result.modelError ? (
          <OutlinedTextArea label="Model error" value={result.modelError} />
        ) : null}
      </div>
    );
  };

  return (
    <div className="zk-demo">
      <div className="zk-header">
        <h3>ZK-LLM Turbo</h3>
        <p>Privacy-preserving split inference with CKKS encrypted layer calls.</p>
      </div>

      <div className="zk-panel">
        <h4>Demo call</h4>
        <p>
          This marketplace demo performs a lightweight health call against the
          hosted service. It verifies the deployed SNET route without requiring
          users to generate CKKS keys or encrypted vectors in the browser.
        </p>
        <StyledButton
          btnText={isRunning ? "Running..." : "Run service check"}
          variant="contained"
          onClick={callHealth}
          disabled={isRunning || isComplete}
        />
      </div>

      <ResultView />
    </div>
  );
};

export default Demo;
