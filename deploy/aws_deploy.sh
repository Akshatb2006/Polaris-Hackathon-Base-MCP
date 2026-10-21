#!/usr/bin/env bash
# One-shot EC2 deployment for the Polaris search MCP server.
#
#   AWS_PROFILE=your-profile REPO_URL=https://github.com/<you>/Polaris-Hackathon-Base-MCP.git \
#   RUNNER_CIDR=0.0.0.0/0 ./deploy/aws_deploy.sh
#
# Creates: one security group (port 8000 from RUNNER_CIDR, 22 from your IP), one EC2
# instance (Amazon Linux 2023, default c7i.large) whose user-data clones the repo with
# Git LFS, builds the Docker image and runs it with the instance's public hostname allowed.
# Prints the MCP URL to submit. Set RUNNER_CIDR to the organizer's published runner IP/32.
set -euo pipefail

REGION="${AWS_REGION:-ap-south-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c7i.large}"
REPO_URL="${REPO_URL:?set REPO_URL to the https clone URL of your fork}"
RUNNER_CIDR="${RUNNER_CIDR:-0.0.0.0/0}"
NAME="${NAME:-polaris-search}"
BRANCH="${BRANCH:-hybrid-search}"
KEY_NAME="${KEY_NAME:-}"

aws() { command aws --region "$REGION" "$@"; }

echo "Account: $(aws sts get-caller-identity --query Account --output text)  Region: $REGION"
AMI=$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
  --query Parameter.Value --output text)
VPC=$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)
MYIP=$(curl -s https://checkip.amazonaws.com)/32

SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values="$NAME" Name=vpc-id,Values="$VPC" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
if [[ -z "$SG" || "$SG" == "None" ]]; then
  SG=$(aws ec2 create-security-group --group-name "$NAME" --description "Polaris MCP search" --vpc-id "$VPC" \
    --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 8000 --cidr "$RUNNER_CIDR" >/dev/null
  [[ "$RUNNER_CIDR" != "$MYIP" ]] && aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp \
    --port 8000 --cidr "$MYIP" >/dev/null 2>&1 || true
  aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP" >/dev/null
fi

USERDATA=$(sed -e "s#__REPO_URL__#$REPO_URL#" -e "s#__BRANCH__#$BRANCH#" "$(dirname "$0")/userdata.sh")

KEY_ARGS=()
[[ -n "$KEY_NAME" ]] && KEY_ARGS=(--key-name "$KEY_NAME")
IID=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" --security-group-ids "$SG" \
  "${KEY_ARGS[@]}" --user-data "$USERDATA" \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=30,VolumeType=gp3}' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
  --query 'Instances[0].InstanceId' --output text)
echo "Instance $IID launching..."
aws ec2 wait instance-running --instance-ids "$IID"
IP=$(aws ec2 describe-instances --instance-ids "$IID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo
echo "Image build takes ~6-10 minutes. Then check:  curl http://$IP:8000/healthz"
echo "Search Lab UI:  http://$IP:8000/"
echo "Submit this MCP endpoint:  http://$IP:8000/mcp"
echo "Smoke test:  python smoke_test.py http://$IP:8000/mcp"
