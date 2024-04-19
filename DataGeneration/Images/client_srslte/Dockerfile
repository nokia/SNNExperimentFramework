FROM srs-zeromq_ue1

ENV http_proxy http://defra1c-proxy.emea.nsn-net.net:8080
ENV https_proxy http://defra1c-proxy.emea.nsn-net.net:8080

ENV DEBIAN_FRONTEND noninteractive
ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get install -y apt-utils

RUN apt-get update \
&& apt-get install -y vlc iproute2 iptables procps net-tools tcpdump tshark git
#RUN apt-get update \
#&& apt-get install -y vlc iproute2 iptables procps net-tools tshark

#RUN snap install vlc

# install ffmpeg from https://trac.ffmpeg.org/ticket/9009
RUN apt-get -y install autoconf automake build-essential libass-dev libfreetype6-dev libgpac-dev \
libsdl1.2-dev libtheora-dev libtool libva-dev libvdpau-dev libvorbis-dev libx11-dev \
libxext-dev libxfixes-dev pkg-config texi2html zlib1g-dev libx264-dev nasm yasm libfaac-dev \
libmp3lame-dev libtheora-dev libvorbis-dev libxvidcore-dev libxext-dev libxfixes-dev
RUN git clone https://github.com/tiamilani/FFmpeg
RUN cd FFmpeg; git branch -a
RUN cd FFmpeg; git checkout remotes/origin/signal
RUN cd FFmpeg; ./configure --enable-gpl --enable-version3 --enable-nonfree --enable-postproc --enable-pthreads --enable-libmp3lame --enable-libtheora --enable-libx264 --enable-libxvid --enable-libvorbis
RUN cd FFmpeg; make -j 8 && make install -j 8 && make distclean && export "PATH=$PATH:$HOME/bin"
#RUN echo "export PATH=/usr/local/ffmpeg/bin:$PATH" >> /root/.bashrc
#RUN echo "export LD_LIBRARY_PATH=/usr/local/ffmpeg/lib:$LD_LIBRARY_PATH" >> /root/.bashrc
#RUN . /root/.bashrc

RUN useradd -m vlcuser

RUN mkdir tmp/result
RUN chown -R vlcuser:vlcuser tmp/

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

COPY acquire_video_stream.sh /tmp/
COPY ffmpeg_acquire.sh /tmp/
COPY tc_set.sh /tmp/
COPY tc_down.sh /tmp/
COPY stream.sdp /tmp/

#ENTRYPOINT ["/tmp/acquire_video_stream.sh"]
