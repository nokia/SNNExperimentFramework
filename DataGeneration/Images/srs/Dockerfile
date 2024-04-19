FROM ubuntu:latest

ENV http_proxy http://defra1c-proxy.emea.nsn-net.net:8080
ENV https_proxy http://defra1c-proxy.emea.nsn-net.net:8080

# Fix timezone issue
ENV TZ=Europe/Rome
ENV PIPPO=pippo
#ARG http_proxy
#ARG https_proxy
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#RUN echo "Acquire::http::Proxy \"${http_proxy}\";\nAcquire::https::Proxy \"${https_proxy}\";" > /etc/apt/apt.conf.d/proxy_bu.conf
#RUN git config --global http.proxy ${http_proxy}

RUN  apt-get update && DEBIAN_FRONTEND=noninteractive apt-get dist-upgrade -y && \
	apt-get install git cmake build-essential libfftw3-dev libmbedtls-dev \
	libboost-program-options-dev libconfig++-dev libsctp-dev libzmq3-dev \
	iperf iperf3 iproute2 net-tools iputils-ping -y && cd && \
	 git clone https://github.com/srsran/srsRAN.git && \
	cd srsRAN && mkdir build && cd build && cmake ../ && make -j `nproc` && make install && cd && \
	rm -rf srsRAN && ldconfig && srsran_install_configs.sh user
COPY ./enb.conf /root/.config/srsran/
COPY ./epc.conf /root/.config/srsran/
COPY ./ue.conf /root/.config/srsran/
COPY ./user_db.csv /root/.config/srsran/
