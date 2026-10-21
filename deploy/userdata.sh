#!/bin/bash
set -eux
dnf install -y docker git
# git-lfs may be missing from the Amazon Linux 2023 repos; fall back to the official release binary.
dnf install -y git-lfs || (curl -fsSL https://github.com/git-lfs/git-lfs/releases/download/v3.6.1/git-lfs-linux-amd64-v3.6.1.tar.gz \
  | tar -xz -C /tmp && /tmp/git-lfs-3.6.1/install.sh)
systemctl enable --now docker
git lfs install --system
cd /opt && git clone -b "__BRANCH__" "__REPO_URL__" app && cd app && git lfs pull
echo "05b6aa5e8e8c1c8c997d584f8288e432862454f8820531e13473443e83c8f073  data/products.json" | sha256sum -c -
TOKEN=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 300')
PUBHOST=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-hostname)
PUBIP=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4)
docker build -t polaris-search .
docker run -d --restart unless-stopped --name polaris-search -p 8000:8000 \
  -e MCP_ALLOWED_HOSTS="localhost:*,127.0.0.1:*,$PUBHOST:*,$PUBHOST,$PUBIP:*,$PUBIP" polaris-search
