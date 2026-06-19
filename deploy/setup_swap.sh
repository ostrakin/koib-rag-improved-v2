#!/usr/bin/env bash
# Создаёт 2 ГБ swap-файл (страховка от OOM при загрузке модели на 2 ГБ ОЗУ).
# Запускать от root: sudo ./setup_swap.sh
set -euo pipefail
SWAP=/swapfile
SIZE_MB=2048

if swapon --show 2>/dev/null | grep -q "$SWAP"; then
  echo "swap уже включён:"; swapon --show; exit 0
fi
echo "Создаю $SWAP размером ${SIZE_MB} МБ ..."
fallocate -l "${SIZE_MB}M" "$SWAP" 2>/dev/null || dd if=/dev/zero of="$SWAP" bs=1M count="$SIZE_MB"
chmod 600 "$SWAP"
mkswap "$SWAP"
swapon "$SWAP"
grep -q "$SWAP" /etc/fstab || echo "$SWAP none swap sw 0 0" >> /etc/fstab
# Меньше выгружать без нужды
echo 'vm.swappiness=10' > /etc/sysctl.d/99-koib-swap.conf
sysctl -p /etc/sysctl.d/99-koib-swap.conf || true
echo "Готово:"; swapon --show; free -h
