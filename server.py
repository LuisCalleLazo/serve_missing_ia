from concurrent import futures
import grpc
from generated import service_pb2_grpc, service_pb2
from generated import model_service_pb2_grpc, model_service_pb2
import DetectionFace.detect_face as detect_face
import PredictionFace.generate_model as generate_model

models = {}

class FaceService(service_pb2_grpc.FaceServiceServicer):


  def AddModel(self, request, context):
      model_id = request.model_id
      model_path = request.model_path
      
      if model_id in models:
          return AddModelResponse(success=False, message="Model already exists.")

      try:
          # Cargar el modelo y asociarlo al ID
          models[model_id] = load_model(model_path)
          return AddModelResponse(success=True, message="Model loaded successfully.")
      except Exception as e:
          return AddModelResponse(success=False, message=str(e))

  def DetectFacePose(self, request, context):
    result = detect_face.detect_pose(request.image_path)
    return service_pb2.DetectFaceResponse(result=result)

  def GenerateModel(self, request, context):
    success = generate_model.create_model(
      request.name_model, request.type_save, request.folder_path
    )
    return service_pb2.GenerateModelResponse(success=success)

def serve():
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  service_pb2_grpc.add_FaceServiceServicer_to_server(FaceService(), server)
  server.add_insecure_port("[::]:5001")
  server.start()
  server.wait_for_termination()

if __name__ == "__main__":
  serve()
