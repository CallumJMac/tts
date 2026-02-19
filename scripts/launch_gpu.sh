#!/usr/bin/env bash
# Launch a spot g5.xlarge for Phase 3.
# Prerequisites: aws cli configured, a key pair, and a security group with SSH (port 22) open.
#
# Usage:
#   ./scripts/launch_gpu.sh              # interactive — prompts for key/sg
#   ./scripts/launch_gpu.sh my-key sg-0abc1234  # non-interactive
#
# After launch, connect with:
#   ssh -i ~/.ssh/<key>.pem ubuntu@<public-ip>

set -euo pipefail

INSTANCE_TYPE="g5.xlarge"
VOLUME_SIZE=100  # GB
REGION="${AWS_DEFAULT_REGION:-${AWS_REGION:-us-east-1}}"

KEY_NAME="${1:-}"
SG_ID="${2:-}"

# --- Prompt for missing args ---
if [[ -z "$KEY_NAME" ]]; then
    echo "Available key pairs in $REGION:"
    aws ec2 describe-key-pairs --region "$REGION" --query 'KeyPairs[*].KeyName' --output table
    echo
    read -rp "Key pair name: " KEY_NAME
fi

if [[ -z "$SG_ID" ]]; then
    echo "Security groups with SSH (port 22) inbound:"
    aws ec2 describe-security-groups --region "$REGION" \
        --filters "Name=ip-permission.from-port,Values=22" \
        --query 'SecurityGroups[*].[GroupId,GroupName]' --output table
    echo
    read -rp "Security group ID (sg-...): " SG_ID
fi

# --- Find the latest Deep Learning Base OSS AMI (Ubuntu 22.04) ---
echo "Looking up Deep Learning AMI in $REGION..."
AMI_ID=$(aws ec2 describe-images --region "$REGION" \
    --owners amazon \
    --filters \
        "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
        "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)

if [[ "$AMI_ID" == "None" || -z "$AMI_ID" ]]; then
    # Fallback: try the broader Deep Learning AMI
    AMI_ID=$(aws ec2 describe-images --region "$REGION" \
        --owners amazon \
        --filters \
            "Name=name,Values=Deep Learning Base GPU AMI (Ubuntu 22.04)*" \
            "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text)
fi

if [[ "$AMI_ID" == "None" || -z "$AMI_ID" ]]; then
    echo "ERROR: Could not find a Deep Learning AMI. Specify one manually."
    exit 1
fi
echo "Using AMI: $AMI_ID"

# --- Check current spot price ---
SPOT_PRICE=$(aws ec2 describe-spot-price-history --region "$REGION" \
    --instance-types "$INSTANCE_TYPE" \
    --product-descriptions "Linux/UNIX" \
    --max-items 1 \
    --query 'SpotPriceHistory[0].SpotPrice' \
    --output text 2>/dev/null || echo "unknown")
echo "Current spot price for $INSTANCE_TYPE in $REGION: \$$SPOT_PRICE/hr"

# --- Launch spot instance ---
echo
echo "Launching spot $INSTANCE_TYPE..."
INSTANCE_ID=$(aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE,\"VolumeType\":\"gp3\"}}]" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tts-phase3}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance launched: $INSTANCE_ID"
echo "Waiting for instance to be running..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo
echo "============================================"
echo "Instance ready!"
echo "  ID:  $INSTANCE_ID"
echo "  IP:  $PUBLIC_IP"
echo "  Cost: ~\$$SPOT_PRICE/hr (spot)"
echo
echo "Connect:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo
echo "Once connected, run:"
echo "  git clone <your-repo-url> tts && cd tts"
echo "  bash scripts/setup_gpu.sh"
echo
echo "When done, terminate:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
echo "============================================"
