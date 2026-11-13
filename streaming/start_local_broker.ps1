param(
    [Parameter(Mandatory=$true)][string]$KafkaHome,
    [Parameter(Mandatory=$true)][string]$JavaHome,
    [string]$StateDir = "$PSScriptRoot/runtime/broker",
    [ValidateRange(1,65535)][int]$Port = 9092,
    [ValidateRange(1,65535)][int]$ControllerPort = 9093
)
$ErrorActionPreference = 'Stop'
if ($Port -eq $ControllerPort) { throw 'Broker and controller ports must differ' }
$java = Join-Path $JavaHome 'bin/java.exe'
$libs = (Join-Path (Resolve-Path $KafkaHome) 'libs/*')
New-Item -ItemType Directory -Force $StateDir | Out-Null
$state = (Resolve-Path $StateDir).Path.Replace('\','/')
$conf = "$state/server.properties"
$settings = @"
process.roles=broker,controller
node.id=1
controller.quorum.bootstrap.servers=127.0.0.1:$ControllerPort
listeners=PLAINTEXT://127.0.0.1:$Port,CONTROLLER://127.0.0.1:$ControllerPort
advertised.listeners=PLAINTEXT://127.0.0.1:$Port,CONTROLLER://127.0.0.1:$ControllerPort
inter.broker.listener.name=PLAINTEXT
controller.listener.names=CONTROLLER
listener.security.protocol.map=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
log.dirs=$state/data
num.partitions=1
offsets.topic.replication.factor=1
transaction.state.log.replication.factor=1
transaction.state.log.min.isr=1
group.initial.rebalance.delay.ms=0
auto.create.topics.enable=false
"@
if ((Test-Path $conf) -and (Get-Content $conf -Raw).Trim() -ne $settings.Trim()) {
    throw 'Broker configuration differs. Use a fresh StateDir for different ports.'
}
$settings | Set-Content -Encoding ascii $conf
if (-not (Test-Path "$state/data/meta.properties")) {
    $id = (& $java -cp $libs kafka.tools.StorageTool random-uuid | Select-Object -Last 1).Trim()
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $java -cp $libs kafka.tools.StorageTool format --standalone -t $id -c $conf
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
$logging = Join-Path (Resolve-Path $KafkaHome) 'config/log4j2.yaml'
& $java -Xms256m -Xmx768m "-Dlog4j2.configurationFile=$logging" "-Dkafka.logs.dir=$state" -cp $libs kafka.Kafka $conf
exit $LASTEXITCODE
