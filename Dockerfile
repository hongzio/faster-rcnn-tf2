FROM tensorflow/tensorflow:2.0.0b1-gpu-py3
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN sed -i 's/#AllowTcpForwarding/AllowTcpForwarding/' /etc/ssh/sshd_config
RUN sed -i 's/#PermitTTY/PermitTTY/' /etc/ssh/sshd_config
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/\/usr\/lib\/openssh\/sftp-server/internal-sftp/' /etc/ssh/sshd_config
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

RUN echo "export PATH=$PATH" >> /etc/profile && echo "ldconfig" >> /etc/profile

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
RUN pip install PyYAML tqdm

RUN rm /etc/bash.bashrc
EXPOSE 22
EXPOSE 6006

ARG ROOT_PW
RUN echo "root:$ROOT_PW" | chpasswd

COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT /entrypoint.sh
