# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: face_service.proto
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
    'face_service.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x12\x66\x61\x63\x65_service.proto\"\'\n\x11\x44\x65tectFaceRequest\x12\x12\n\nimage_path\x18\x01 \x01(\t\"$\n\x12\x44\x65tectFaceResponse\x12\x0e\n\x06result\x18\x01 \x01(\t2H\n\x0b\x46\x61\x63\x65Service\x12\x39\n\x0e\x44\x65tectFacePose\x12\x12.DetectFaceRequest\x1a\x13.DetectFaceResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'face_service_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_DETECTFACEREQUEST']._serialized_start=22
  _globals['_DETECTFACEREQUEST']._serialized_end=61
  _globals['_DETECTFACERESPONSE']._serialized_start=63
  _globals['_DETECTFACERESPONSE']._serialized_end=99
  _globals['_FACESERVICE']._serialized_start=101
  _globals['_FACESERVICE']._serialized_end=173
# @@protoc_insertion_point(module_scope)