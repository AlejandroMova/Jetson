FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch

ENV DEBIAN_FRONTEND=noninteractive

# descargar setup de ngc
RUN wget --content-disposition \
    https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/4.13.0/files/ngccli_linux.zip \
    -O ngccli_linux.zip \
    && unzip ngccli_linux.zip \
    && rm ngccli_linux.zip \
    && mv ngc-cli /opt/ngc-cli \
    && chmod +x /opt/ngc-cli/ngc \
    && ln -s /opt/ngc-cli/ngc /usr/local/bin/ngc

# python bindings a deepstream
RUN apt-get update && apt-get install -y \
    python3-gi python3-dev python3-gst-1.0 python-gi-dev git meson \
    python3 python3-pip python3-venv cmake g++ build-essential libglib2.0-dev \
    libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf automake libgirepository-2.0-dev libcairo2-dev

RUN pip install build

WORKDIR /opt/nvidia/deepstream/deepstream/sources/

RUN git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git

WORKDIR /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/

RUN git submodule update --init

RUN python3 bindings/3rdparty/git-partial-submodule/git-partial-submodule.py restore-sparse

WORKDIR /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings/3rdparty/gstreamer/subprojects/gst-python/

RUN meson setup build 

WORKDIR /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings/3rdparty/gstreamer/subprojects/gst-python/build/

RUN ninja && ninja install 

WORKDIR /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings/
RUN export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) && python3 -m build 

WORKDIR /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings/dist/

RUN pip3 install ./pyds-1.2.2-*.whl

WORKDIR /nx_tech/

COPY requirements.txt /nx_tech/

RUN pip3 install --no-cache-dir -r requirements.txt

# COPY . /nx_tech/

CMD ["/bin/bash"]

# docker run -it --rm --gpus all \
#   --shm-size=16g \
#   -v /mnt/windows/Users/alexm/Desktop/NX_tech:/workspace/nx_tech \
#   -w /workspace/nx_tech \
#   nvcr.io/nvidia/tao/tao-toolkit:6.25.11-pyt \
#   /bin/bash