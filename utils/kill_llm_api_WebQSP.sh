for port in 8101 8102; do
  pid=$(lsof -t -i:$port)
  if [ ! -z "$pid" ]; then
    echo "Killing process $pid on port $port..."
    kill -9 $pid
  else
    echo "No process found on port $port."
  fi
done