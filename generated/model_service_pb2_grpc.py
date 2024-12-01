# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

import model_service_pb2 as model__service__pb2

GRPC_GENERATED_VERSION = '1.68.0'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in model_service_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class ModelServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.AddModel = channel.unary_unary(
                '/model.ModelService/AddModel',
                request_serializer=model__service__pb2.AddModelRequest.SerializeToString,
                response_deserializer=model__service__pb2.AddModelResponse.FromString,
                _registered_method=True)
        self.RemoveModel = channel.unary_unary(
                '/model.ModelService/RemoveModel',
                request_serializer=model__service__pb2.RemoveModelRequest.SerializeToString,
                response_deserializer=model__service__pb2.RemoveModelResponse.FromString,
                _registered_method=True)
        self.AnalyzeImage = channel.unary_unary(
                '/model.ModelService/AnalyzeImage',
                request_serializer=model__service__pb2.AnalyzeImageRequest.SerializeToString,
                response_deserializer=model__service__pb2.AnalyzeImageResponse.FromString,
                _registered_method=True)
        self.ListModels = channel.unary_unary(
                '/model.ModelService/ListModels',
                request_serializer=model__service__pb2.Empty.SerializeToString,
                response_deserializer=model__service__pb2.ListModelsResponse.FromString,
                _registered_method=True)


class ModelServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def AddModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RemoveModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def AnalyzeImage(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ListModels(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ModelServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'AddModel': grpc.unary_unary_rpc_method_handler(
                    servicer.AddModel,
                    request_deserializer=model__service__pb2.AddModelRequest.FromString,
                    response_serializer=model__service__pb2.AddModelResponse.SerializeToString,
            ),
            'RemoveModel': grpc.unary_unary_rpc_method_handler(
                    servicer.RemoveModel,
                    request_deserializer=model__service__pb2.RemoveModelRequest.FromString,
                    response_serializer=model__service__pb2.RemoveModelResponse.SerializeToString,
            ),
            'AnalyzeImage': grpc.unary_unary_rpc_method_handler(
                    servicer.AnalyzeImage,
                    request_deserializer=model__service__pb2.AnalyzeImageRequest.FromString,
                    response_serializer=model__service__pb2.AnalyzeImageResponse.SerializeToString,
            ),
            'ListModels': grpc.unary_unary_rpc_method_handler(
                    servicer.ListModels,
                    request_deserializer=model__service__pb2.Empty.FromString,
                    response_serializer=model__service__pb2.ListModelsResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'model.ModelService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('model.ModelService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class ModelService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def AddModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/model.ModelService/AddModel',
            model__service__pb2.AddModelRequest.SerializeToString,
            model__service__pb2.AddModelResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def RemoveModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/model.ModelService/RemoveModel',
            model__service__pb2.RemoveModelRequest.SerializeToString,
            model__service__pb2.RemoveModelResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def AnalyzeImage(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/model.ModelService/AnalyzeImage',
            model__service__pb2.AnalyzeImageRequest.SerializeToString,
            model__service__pb2.AnalyzeImageResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def ListModels(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/model.ModelService/ListModels',
            model__service__pb2.Empty.SerializeToString,
            model__service__pb2.ListModelsResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)