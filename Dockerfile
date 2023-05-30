# to build an image build run the following command:
# docker build --rm=true -t psu-postgresql .
# to build a container run the following command once the image is created:
# docker run -d -p 49168:5432 psu-postgresql
# once the container is created run the following command to get the id of the container
# docker ps -a
# to start a container run the following command:
# docker start <containerid>
# to access a container at bash prompt run the following command:
# docker exec -it <containerid> bash
# to stop a container run the following command:
# docker stop <containerid>
# to see the log in a container run the following command:
# docker logs <containerid>

FROM psu-ubuntu-java:latest

# make sure we work as root
USER root

# set environment variables
ENV POSTGRESQL_HOME /usr/lib/postgresql/12
ENV LD_LIBRARY_PATH $POSTGRESQL_HOME/lib
ENV PATH $POSTGRESQL_HOME/bin:$PATH
ENV MANPATH=$POSTGRESQL_HOME/share/man:$MANPATH
ENV PGDATA $POSTGRESQL_HOME/data

# do an update and install prerequisites
RUN \
   apt-get update && \
   apt-get install -y \
      apt-utils \
      acl \
      gcc \
      libperl-dev \
      make \
      wget \
      gpg-agent \
      software-properties-common \
      locales && \
      rm -rf /var/lib/apt/lists/* && \
      localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

ENV LANG en_US.utf8
ENV TZ=America/New_York
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn
ENV DEBIAN_FRONTEND=noninteractive 
# Install PostgreSQL 
RUN \
  wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -  && \
  echo "deb http://apt.postgresql.org/pub/repos/apt/ `lsb_release -cs`-pgdg main" | tee  /etc/apt/sources.list.d/pgdg.list && \
  apt-get update && \
  apt-get -yq install postgresql-12 postgresql-client-12 && \
  echo "#!/bin/bash" >> /startpostgres.sh && \
  echo "pg_ctlcluster 12 main start" >> /startpostgres.sh && \
  echo "tail -f /dev/null" >> /startpostgres.sh && \
  chmod a+wx /startpostgres.sh
  
# make sure we work as postgres user
USER postgres

# do postgresql stuff   
# create pg user student with password student and a database student
RUN \
   pg_ctlcluster 12 main start && \
   echo "host all  all    0.0.0.0/0  md5" >> /etc/postgresql/12/main/pg_hba.conf && \
   echo "listen_addresses = '*'" >> /etc/postgresql/12/main/postgresql.conf && \
   /usr/bin/psql --command "CREATE USER student WITH SUPERUSER PASSWORD 'student';" && \
   /usr/lib/postgresql/12/bin/createdb -O student student
   
# Expose the PostgreSQL port
EXPOSE 5432

ENTRYPOINT ["/startpostgres.sh"]

#/usr/bin/psql --command "SELECT * FROM pg_catalog.pg_tables;" 