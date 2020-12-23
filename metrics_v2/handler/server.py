import os

import functions_framework
import functions_framework._http

from absl import flags
from handler import main

def serve(argv):
  app = functions_framework.create_app('receive_test_event', main.__file__, 'cloudevent')
  functions_framework._http.create_server(
      app, debug=False).run(os.getenv('HOST', '0.0.0.0'), os.getenv('PORT', 8080))

if __name__ == "__main__":
  main.define_flags()
  app.run(serve)
