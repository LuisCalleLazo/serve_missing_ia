from concurrent import futures
import grpc
from generated import service_pb2_grpc, service_pb2
import DetectionFace.detect_face_pose as detect_face_pose
import PredictionFace.generate_model as generate_model
import sys
import signal

models = {}

class FaceService(service_pb2_grpc.FaceServiceServicer):

  # def AddModel(self, request, context):
  #   model_id = request.model_id
  #   model_path = request.model_path
    
  #   if model_id in models:
  #     return AddModelResponse(success=False, message="Model already exists.")

  #   try:
  #     # Cargar el modelo y asociarlo al ID
  #     models[model_id] = load_model(model_path)
  #     return AddModelResponse(success=True, message="Model loaded successfully.")
  #   except Exception as e:
  #     return AddModelResponse(success=False, message=str(e))

  def DetectFacePose(self, request, context):
    result = detect_face_pose.detect_face_pose(request.image_path)
    return service_pb2.DetectFaceResponse(result=result)

  def GenerateModel(self, request, context):
    success = generate_model.generate_model(
      request.name_model, request.type_save, request.folder_path
    )
    return service_pb2.GenerateModelResponse(success=success)

def serve(port):
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  service_pb2_grpc.add_FaceServiceServicer_to_server(FaceService(), server)
  server.add_insecure_port(f"[::]:{port}")
  server.start()


  # Manejo de señal para detener el servidor
  def signal_handler(sig, frame):
    print("Shutting down gracefully...")
    server.stop(0)
    sys.exit(0)


  signal.signal(signal.SIGINT, signal_handler)
  server.wait_for_termination()

if __name__ == "__main__":
  serve(5001)
