
from generated import face_service_pb2_grpc, face_service_pb2
import DetectionFace.detect_face_pose as detect_face_pose

class FaceService(face_service_pb2_grpc.FaceServiceServicer):

  def DetectFacePose(self, request, context):
    result = detect_face_pose.detect_face_pose(request.image_path)
    return face_service_pb2.DetectFaceResponse(result=result)