# Docker Environment
Build from the root folder of the repository the docker container. Run:
```
docker build ./docker/ -t elmo-bilstm
```

This builds the Python 3.6 container and assigns the name *bilstm* to it. 

To run our code, we first must start the container and mount the current folder into the container. On Linux, you can use the variable $PWD to get your current path. On Windows, you can use the `%cd` command.
```
docker run -it -v "$PWD":/src elmo-bilstm bash
```

Alternatively, you can mount any other folder to be used within your container:

```
docker run -it -v /my/path/for/code:/src elmo-bilstm bash
```


