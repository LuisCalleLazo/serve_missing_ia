# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: model_service.proto
# Protobuf Python Version: 5.28.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    28,
    1,
    '',
    'model_service.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x13model_service.proto\x12\x05model\"8\n\x0f\x41\x64\x64ModelRequest\x12\x11\n\tperson_id\x18\x01 \x01(\x05\x12\x12\n\nmodel_path\x18\x02 \x01(\t\"4\n\x10\x41\x64\x64ModelResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t\"\'\n\x12RemoveModelRequest\x12\x11\n\tperson_id\x18\x01 \x01(\x05\"7\n\x13RemoveModelResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t\"<\n\x13\x41nalyzeImageRequest\x12\x11\n\tperson_id\x18\x01 \x01(\x05\x12\x12\n\nimage_data\x18\x02 \x01(\x0c\"C\n\x14\x41nalyzeImageResponse\x12\x1a\n\x12is_person_detected\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t\"6\n\x12ListModelsResponse\x12 \n\x06models\x18\x01 \x03(\x0b\x32\x10.model.ModelInfo\"2\n\tModelInfo\x12\x11\n\tperson_id\x18\x01 \x01(\x05\x12\x12\n\nmodel_path\x18\x02 \x01(\t\"\x07\n\x05\x45mpty2\x91\x02\n\x0cModelService\x12;\n\x08\x41\x64\x64Model\x12\x16.model.AddModelRequest\x1a\x17.model.AddModelResponse\x12\x44\n\x0bRemoveModel\x12\x19.model.RemoveModelRequest\x1a\x1a.model.RemoveModelResponse\x12G\n\x0c\x41nalyzeImage\x12\x1a.model.AnalyzeImageRequest\x1a\x1b.model.AnalyzeImageResponse\x12\x35\n\nListModels\x12\x0c.model.Empty\x1a\x19.model.ListModelsResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'model_service_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_ADDMODELREQUEST']._serialized_start=30
  _globals['_ADDMODELREQUEST']._serialized_end=86
  _globals['_ADDMODELRESPONSE']._serialized_start=88
  _globals['_ADDMODELRESPONSE']._serialized_end=140
  _globals['_REMOVEMODELREQUEST']._serialized_start=142
  _globals['_REMOVEMODELREQUEST']._serialized_end=181
  _globals['_REMOVEMODELRESPONSE']._serialized_start=183
  _globals['_REMOVEMODELRESPONSE']._serialized_end=238
  _globals['_ANALYZEIMAGEREQUEST']._serialized_start=240
  _globals['_ANALYZEIMAGEREQUEST']._serialized_end=300
  _globals['_ANALYZEIMAGERESPONSE']._serialized_start=302
  _globals['_ANALYZEIMAGERESPONSE']._serialized_end=369
  _globals['_LISTMODELSRESPONSE']._serialized_start=371
  _globals['_LISTMODELSRESPONSE']._serialized_end=425
  _globals['_MODELINFO']._serialized_start=427
  _globals['_MODELINFO']._serialized_end=477
  _globals['_EMPTY']._serialized_start=479
  _globals['_EMPTY']._serialized_end=486
  _globals['_MODELSERVICE']._serialized_start=489
  _globals['_MODELSERVICE']._serialized_end=762
# @@protoc_insertion_point(module_scope)
