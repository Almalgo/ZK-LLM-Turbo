// package: zk_llm
// file: zk_llm_http_api.proto

var zk_llm_http_api_pb = require("./zk_llm_http_api_pb");
var grpc = require("@improbable-eng/grpc-web").grpc;

var ZKLLMService = (function () {
  function ZKLLMService() {}
  ZKLLMService.serviceName = "zk_llm.ZKLLMService";
  return ZKLLMService;
}());

ZKLLMService.Health = {
  methodName: "Health",
  service: ZKLLMService,
  requestStream: false,
  responseStream: false,
  requestType: zk_llm_http_api_pb.HealthRequest,
  responseType: zk_llm_http_api_pb.HealthResponse
};

ZKLLMService.Session = {
  methodName: "Session",
  service: ZKLLMService,
  requestStream: false,
  responseStream: false,
  requestType: zk_llm_http_api_pb.SessionRequest,
  responseType: zk_llm_http_api_pb.SessionResponse
};

ZKLLMService.Layer = {
  methodName: "Layer",
  service: ZKLLMService,
  requestStream: false,
  responseStream: false,
  requestType: zk_llm_http_api_pb.LayerRequest,
  responseType: zk_llm_http_api_pb.LayerResponse
};

exports.ZKLLMService = ZKLLMService;

function ZKLLMServiceClient(serviceHost, options) {
  this.serviceHost = serviceHost;
  this.options = options || {};
}

ZKLLMServiceClient.prototype.health = function health(requestMessage, metadata, callback) {
  if (arguments.length === 2) {
    callback = arguments[1];
  }
  var client = grpc.unary(ZKLLMService.Health, {
    request: requestMessage,
    host: this.serviceHost,
    metadata: metadata,
    transport: this.options.transport,
    debug: this.options.debug,
    onEnd: function (response) {
      if (callback) {
        if (response.status !== grpc.Code.OK) {
          var err = new Error(response.statusMessage);
          err.code = response.status;
          err.metadata = response.trailers;
          callback(err, null);
        } else {
          callback(null, response.message);
        }
      }
    }
  });
  return {
    cancel: function () {
      callback = null;
      client.close();
    }
  };
};

ZKLLMServiceClient.prototype.session = function session(requestMessage, metadata, callback) {
  if (arguments.length === 2) {
    callback = arguments[1];
  }
  var client = grpc.unary(ZKLLMService.Session, {
    request: requestMessage,
    host: this.serviceHost,
    metadata: metadata,
    transport: this.options.transport,
    debug: this.options.debug,
    onEnd: function (response) {
      if (callback) {
        if (response.status !== grpc.Code.OK) {
          var err = new Error(response.statusMessage);
          err.code = response.status;
          err.metadata = response.trailers;
          callback(err, null);
        } else {
          callback(null, response.message);
        }
      }
    }
  });
  return {
    cancel: function () {
      callback = null;
      client.close();
    }
  };
};

ZKLLMServiceClient.prototype.layer = function layer(requestMessage, metadata, callback) {
  if (arguments.length === 2) {
    callback = arguments[1];
  }
  var client = grpc.unary(ZKLLMService.Layer, {
    request: requestMessage,
    host: this.serviceHost,
    metadata: metadata,
    transport: this.options.transport,
    debug: this.options.debug,
    onEnd: function (response) {
      if (callback) {
        if (response.status !== grpc.Code.OK) {
          var err = new Error(response.statusMessage);
          err.code = response.status;
          err.metadata = response.trailers;
          callback(err, null);
        } else {
          callback(null, response.message);
        }
      }
    }
  });
  return {
    cancel: function () {
      callback = null;
      client.close();
    }
  };
};

exports.ZKLLMServiceClient = ZKLLMServiceClient;

