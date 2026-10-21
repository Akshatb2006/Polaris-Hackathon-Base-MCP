#!/usr/bin/env bash
# Remove what aws_deploy.sh created: instances tagged Name=$NAME and the security group $NAME.
#
#   AWS_PROFILE=your-profile AWS_REGION=ap-south-1 NAME=polaris-search ./deploy/aws_teardown.sh
#
# Lists everything first and asks for confirmation (YES=1 skips). Root volumes are deleted with
# their instances. Nothing else in the account is touched.
set -euo pipefail

REGION="${AWS_REGION:-ap-south-1}"
NAME="${NAME:-polaris-search}"
aws() { command aws --region "$REGION" "$@"; }

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
IIDS=$(aws ec2 describe-instances --filters Name=tag:Name,Values="$NAME" \
  Name=instance-state-name,Values=pending,running,stopping,stopped \
  --query 'Reservations[].Instances[].InstanceId' --output text)
SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values="$NAME" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
[[ "$SG" == "None" ]] && SG=""

echo "Account $ACCOUNT, region $REGION"
echo "  instances to terminate:     ${IIDS:-none}"
echo "  security group to delete:   ${SG:-none}"
[[ -n "$IIDS$SG" ]] || { echo "Nothing to remove."; exit 0; }
if [[ "${YES:-}" != "1" ]]; then
  read -r -p "Delete these permanently? [y/N] " answer
  [[ "$answer" == "y" || "$answer" == "Y" ]] || { echo "cancelled; nothing was deleted"; exit 1; }
fi

if [[ -n "$IIDS" ]]; then
  # shellcheck disable=SC2086
  aws ec2 terminate-instances --instance-ids $IIDS >/dev/null
  echo "Waiting for termination..."
  # shellcheck disable=SC2086
  aws ec2 wait instance-terminated --instance-ids $IIDS
fi
if [[ -n "$SG" ]]; then
  aws ec2 delete-security-group --group-id "$SG"
fi
echo "Removed."
