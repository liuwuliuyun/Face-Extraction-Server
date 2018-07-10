# FaceExtractionServer

ZEROMQ implementation

Yun Liu (liuwuliuyun@qq.com)

### Notion: current version can only run in docker environment!

### Requirements:

python >= 3.6

Tensorflow(GPU Version) for python 3 >= 1.7.0

zeromq latest

numpy latest

flask latest

### Architecture

```

                worker_0.py
                
  server.py ->              -> embed_collector.py
  
                worker_1.py
```
