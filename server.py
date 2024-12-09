from concurrent import futures
import grpc
import sys
import signal

from generated import face_service_pb2_grpc, predict_service_pb2_grpc
from Services.face_service import FaceService
from Services.predict_face_service import PredictService

models = {}

def LoadModels():
  pass


def serve(port):
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  
  face_service_pb2_grpc.add_FaceServiceServicer_to_server(FaceService(), server)
  predict_service_pb2_grpc.add_PredictServiceServicer_to_server(PredictService(), server)

  server.add_insecure_port(f"[::]:{port}")
  server.start()

  def signal_handler(sig, frame):
    print("Shutting down gracefully...")
    server.stop(0)
    sys.exit(0)


  signal.signal(signal.SIGINT, signal_handler)
  server.wait_for_termination()

if __name__ == "__main__":
  serve(5001)