FROM srs-zeromq_epc

ENV http_proxy http://defra1c-proxy.emea.nsn-net.net:8080
ENV https_proxy http://defra1c-proxy.emea.nsn-net.net:8080

ENV DEBIAN_FRONTEND noninteractive
ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#RUN apt-get update \
#&& apt-get install -y software-properties-common

#RUN add-apt-repository ppa:videolan/stable-daily

RUN apt-get update \
&& apt-get install -y vlc ffmpeg iproute2 iptables iputils-ping net-tools nload

RUN apt-get install -y procps tcpdump tshark

RUN useradd -m vlcuser

RUN groupadd pcap
RUN usermod -a -G pcap vlcuser
RUN chgrp pcap /usr/bin/tcpdump
RUN chmod 750 /usr/bin/tcpdump
RUN groupadd wireshark
RUN usermod -a -G wireshark vlcuser
RUN chgrp pcap /usr/bin/dumpcap
RUN chmod 750 /usr/bin/dumpcap
RUN setcap cap_net_raw,cap_net_admin=eip /usr/bin/tcpdump
RUN setcap cap_net_raw,cap_net_admin=eip /usr/bin/dumpcap


COPY simulate_video_stream.sh /tmp/
COPY ffmpeg_stream.sh /tmp/
COPY generate_sdp.sh /tmp/
COPY tc_set.sh /tmp/
COPY tc_down.sh /tmp/

RUN mkdir tmp/result
RUN chown -R vlcuser:vlcuser tmp/*
#ENTRYPOINT ["/tmp/simulate_video_stream.sh"]
