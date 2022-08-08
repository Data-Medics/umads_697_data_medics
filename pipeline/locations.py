import os

base_dir = os.path.dirname(os.path.dirname(__file__))
data = os.path.join(base_dir, "data", "HumAID_data_v1.0")
blog_data = os.path.join(base_dir, "blog_post", "blog_data")
outputs = os.path.join(base_dir, "pipeline", "output")
