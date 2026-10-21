#!/usr/bin/env bash
# One-shot EC2 deployment for the Polaris search MCP server.
#
#   AWS_PROFILE=your-profile AWS_REGION=ap-south-1 \
#   REPO_URL=https://github.com/<you>/Polaris-Hackathon-Base-MCP.git BRANCH=hybrid-search COMMIT=79f46e4 \
#   RUNNER_CIDR=<runner-ip>/32 ./deploy/aws_deploy.sh
#
# Creates exactly two billable/visible resources, both tagged Project=polaris-search:
#   - security group "$NAME": TCP 8000 from RUNNER_CIDR and from your current IP;
#     TCP 22 from your IP only when KEY_NAME is set
#   - one EC2 instance (Amazon Linux 2023 x86_64, default c7i.large, 30 GB gp3 root volume that
#     is deleted with the instance) whose user-data clones the repo with Git LFS, verifies the
#     catalog checksum, builds the Docker image and runs it.
# Asks for confirmation before creating anything (set YES=1 to skip). Refuses to launch a second
# instance while one tagged with NAME is pending or running. Remove everything with
# ./deploy/aws_teardown.sh.
#
# The allowed Host header is fixed at first boot. Stopping and starting the instance changes its
# public IP and breaks the endpoint (HTTP 421); reboot is fine. To move, tear down and redeploy.
set -euo pipefail

REGION="${AWS_REGION:-ap-south-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c7i.large}"
REPO_URL="${REPO_URL:?set REPO_URL to the https clone URL of your fork}"
RUNNER_CIDR="${RUNNER_CIDR:-0.0.0.0/0}"
NAME="${NAME:-polaris-search}"
BRANCH="${BRANCH:-hybrid-search}"
COMMIT="${COMMIT:-}"
KEY_NAME="${KEY_NAME:-}"
TAGS="{Key=Name,Value=$NAME},{Key=Project,Value=polaris-search}"

aws() { command aws --region "$REGION" "$@"; }
die() { echo "error: $*" >&2; exit 1; }

ACCOUNT=$(aws sts get-caller-identity --query Account --output text) || die "AWS credentials not usable (AWS_PROFILE=${AWS_PROFILE:-default})"
VPC=$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)
[[ "$VPC" != "None" && -n "$VPC" ]] || die "no default VPC in $REGION; create one with: aws ec2 create-default-vpc --region $REGION"
OFFERED=$(aws ec2 describe-instance-type-offerings --filters Name=instance-type,Values="$INSTANCE_TYPE" \
  --query 'length(InstanceTypeOfferings)' --output text)
[[ "$OFFERED" != "0" ]] || die "$INSTANCE_TYPE is not offered in $REGION; set INSTANCE_TYPE (x86_64, >= 4 GiB RAM)"
EXISTING=$(aws ec2 describe-instances --filters Name=tag:Name,Values="$NAME" Name=instance-state-name,Values=pending,running \
  --query 'Reservations[].Instances[].InstanceId' --output text)
[[ -z "$EXISTING" ]] || die "instance(s) $EXISTING named $NAME already running; tear down first or set NAME"
AMI=$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
  --query Parameter.Value --output text)
MYIP="$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')/32"

echo "About to create in account $ACCOUNT, region $REGION:"
echo "  security group  $NAME  (port 8000 from $RUNNER_CIDR and $MYIP${KEY_NAME:+; port 22 from $MYIP})"
echo "  instance        $INSTANCE_TYPE  $AMI  30 GB gp3  key=${KEY_NAME:-none}"
echo "  code            $REPO_URL  branch $BRANCH${COMMIT:+  commit $COMMIT}"
[[ "$RUNNER_CIDR" == "0.0.0.0/0" ]] && echo "  WARNING: port 8000 will be open to the whole internet (set RUNNER_CIDR to the runner IP/32)"
echo "  Billing runs until ./deploy/aws_teardown.sh is executed."
if [[ "${YES:-}" != "1" ]]; then
  read -r -p "Proceed? [y/N] " answer
  [[ "$answer" == "y" || "$answer" == "Y" ]] || die "cancelled; nothing was created"
fi

SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values="$NAME" Name=vpc-id,Values="$VPC" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
if [[ -z "$SG" || "$SG" == "None" ]]; then
  SG=$(aws ec2 create-security-group --group-name "$NAME" --description "Polaris MCP search" --vpc-id "$VPC" \
    --tag-specifications "ResourceType=security-group,Tags=[$TAGS]" --query GroupId --output text)
fi
allow() {  # idempotent: an existing identical rule is not an error
  aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port "$1" --cidr "$2" >/dev/null 2>&1 \
    || aws ec2 describe-security-groups --group-ids "$SG" --query "SecurityGroups[0].IpPermissions[?ToPort==\`$1\`].IpRanges[].CidrIp" \
       --output text | tr '\t' '\n' | grep -qx "$2" || die "could not open port $1 for $2"
}
allow 8000 "$RUNNER_CIDR"
[[ "$RUNNER_CIDR" == "$MYIP" ]] || allow 8000 "$MYIP"
[[ -z "$KEY_NAME" ]] || allow 22 "$MYIP"

USERDATA=$(sed -e "s#__REPO_URL__#$REPO_URL#" -e "s#__BRANCH__#$BRANCH#" -e "s#__COMMIT__#$COMMIT#" \
  "$(dirname "$0")/userdata.sh")
KEY_ARGS=()
[[ -z "$KEY_NAME" ]] || KEY_ARGS=(--key-name "$KEY_NAME")
# ${arr[@]+...} keeps an empty array legal under `set -u` on macOS bash 3.2
IID=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" --security-group-ids "$SG" \
  ${KEY_ARGS[@]+"${KEY_ARGS[@]}"} --user-data "$USERDATA" \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=30,VolumeType=gp3,DeleteOnTermination=true}' \
  --metadata-options 'HttpTokens=required,HttpEndpoint=enabled' \
  --tag-specifications "ResourceType=instance,Tags=[$TAGS]" "ResourceType=volume,Tags=[$TAGS]" \
  --query 'Instances[0].InstanceId' --output text)
echo "Instance $IID launching..."
aws ec2 wait instance-running --instance-ids "$IID"
IP=$(aws ec2 describe-instances --instance-ids "$IID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo
echo "Instance $IID at $IP. The image build takes about 6-10 minutes after boot."
echo "Verify from this machine:   ./deploy/verify_endpoint.sh http://$IP:8000"
echo "Search Lab UI:              http://$IP:8000/"
echo "Endpoint to submit:         http://$IP:8000/mcp"
echo "Tear down when finished:    AWS_PROFILE=${AWS_PROFILE:-default} AWS_REGION=$REGION NAME=$NAME ./deploy/aws_teardown.sh"
