# 首次安装与连接准备

本文只说明源码仓库对应的首次准备和可验证检查。PostgreSQL 服务、角色和数据库可能需要管理员准备；QGIS profile 和用户目录下的插件安装通常由当前用户完成。本文没有替代目标机器的权限或认证策略。

## 1. 先确定 QGIS 与 profile

当前安装脚本要求 QGIS 4.2.x，并将插件放入所选 QGIS4 profile 的 `python/plugins/`。脚本默认使用名为 `default` 的 profile，这只是脚本默认值，不代表当前 QGIS 正在使用它。

在目标 QGIS 中打开“当前用户 profile 文件夹”或通过 QGIS 的用户 profile 管理界面确认正在使用的 profile 名称。也可以先查看脚本对应的 profile 根目录：

```text
Ubuntu: ~/.local/share/QGIS/QGIS4/profiles/
macOS:  ~/Library/Application Support/QGIS/QGIS4/profiles/
```

选择当前 QGIS 实际使用的目录名，例如 `QGIS4`，然后把同一个名称传给安装脚本：

```bash
scripts/deploy/install_plugin.sh --platform ubuntu --profile QGIS4
scripts/deploy/install_plugin.sh --platform macos --profile QGIS4
```

上面的两行是平台示例，按目标平台执行其中一行。若 profile 目录不在脚本默认位置，使用 QGIS 已确认的完整 `python/plugins` 目录传入 `--plugin-dir`；不要仅凭 `default` 或历史部署记录判断目标。脚本也支持 `--platform auto`，它按当前操作系统选择 `ubuntu` 或 `macos`。

## 2. 准备 PostgreSQL 角色和数据库

插件默认连接的是本机 PostgreSQL Unix socket，数据库名和角色名都取当前系统用户名：

```text
dbname=<当前系统用户名> user=<当前系统用户名> host=/var/run/postgresql port=5432
```

macOS 上，如果 `/var/run/postgresql` 不存在且 PostgreSQL 在 `/tmp/.s.PGSQL.5432` 提供 socket，运行时会使用 `/tmp`。默认连接依赖目标 PostgreSQL 已存在同名角色和数据库，并且该角色能通过目标主机的本地认证规则连接；安装 PostgreSQL 软件本身不会自动满足这两个条件。

如果管理员确认同名角色和数据库已经存在，先只做连接检查（不会创建或修改数据库）：

```bash
psql --version
psql -d "$USER" -U "$USER" -h /var/run/postgresql -p 5432 \
  -c 'select current_user, current_database(), version();'
```

在 macOS 上若 `/var/run/postgresql` 不存在，使用：

```bash
psql -d "$USER" -U "$USER" -h /tmp -p 5432 \
  -c 'select current_user, current_database(), version();'
```

若 `psql` 报角色不存在、数据库不存在、socket 不存在或认证失败，先把错误交给 PostgreSQL 管理员处理。管理员需要按本机认证策略准备一个可连接的角色和数据库；角色名、数据库名可以继续使用当前系统用户名，也可以使用管理员批准的专用名称。若采用专用名称，不要把密码写进仓库或命令历史，改用管理员批准的环境变量、`.pgpass` 或 libpq service，并通过 `LOESS_STATE_DB_DSN` 指定完整 DSN。

只有在管理员确认目标角色或数据库确实缺失、并提供管理连接参数后，才按实际环境填写下面的模板。模板只展示参数位置，不提供猜测的 `sudo`、主机、端口或管理账号；执行者应先用 `createuser --help` 和 `createdb --help` 核对本机版本，并由管理员执行或监督：

```bash
createuser --host='<管理连接地址或 socket 目录>' --port='<管理端口>' \
  --username='<管理角色>' --login --no-createdb --no-createrole \
  --no-superuser '<应用角色>'
createdb --host='<管理连接地址或 socket 目录>' --port='<管理端口>' \
  --username='<管理角色>' --owner='<应用角色>' '<数据库名>'
```

准备完成后，目标 `<role>` 必须能够登录 `<database>`，并能创建或使用项目 schema，以及读写该 schema 中运行所需的表。若管理员采用专用 schema 或更细的权限拆分，应在连接检查前明确授予这些权限；不要用超级用户权限代替项目角色。

使用替代 DSN 时，先在当前 shell 做连接检查，再启动 QGIS：

```bash
export LOESS_STATE_DB_DSN='dbname=<database> user=<role> host=<socket-or-host> port=5432'
export LOESS_STATE_DB_SCHEMA='loess_qgis'
psql "$LOESS_STATE_DB_DSN" \
  -c 'select current_user, current_database(), version();'
```

以上示例中的尖括号值必须由管理员提供。需要密码时使用目标平台的 libpq 认证配置；不要将密码放入 DSN、README、shell 历史或日志。这个 `export` 只对继承当前 shell 环境的进程有效；连接检查通过后，应从同一终端启动实际的 QGIS 可执行文件，或在同一环境中重启目标 QGIS 实例。通过 macOS Dock/Finder 启动的 QGIS 不应假定会继承该 shell 变量。首次连接检查只证明登录、数据库和服务器可达，不等于已经完成插件运行验收。

## 3. 初始化运行项目并检查资产

在确认 Conda、PostgreSQL 和目标 QGIS 已由管理员准备后，按根 README 的 `init_project.sh` 示例初始化运行项目。需要核对资产时可先使用 `--check-only --check-assets`；该检查只验证源码和目标项目中的资产状态，不能代替 PostgreSQL 连接检查。

## 4. 安装后核对 profile、QGIS 和插件版本

安装脚本会先检测 QGIS 4.2.x，再输出实际安装目录和 Git SHA。安装完成后，在输出的目录执行只读核对：

```bash
PLUGIN_DIR='<安装脚本输出的完整插件目录>'
grep -E '^(name|version|qgisMinimumVersion|qgisMaximumVersion)=' \
  "$PLUGIN_DIR/metadata.txt"
grep -E '"(plugin_version|qgis_profile|platform)"' \
  "$PLUGIN_DIR/deployment_manifest.json"
```

插件版本以源码 `src/labeling_tool/metadata.txt` 为准，安装脚本从该文件读取版本并写入部署清单。核对安装目录中的 `version`、清单中的 `plugin_version` 与脚本输出是否一致，同时确认所选 QGIS profile 和目标平台。版本号本身不代表已经完成实机验收。随后重启该 profile 的 QGIS，在插件管理器或插件菜单中确认“地物标注工具”可加载，再进行单独的真实输入验收。

## 5. 常见结果的处理边界

- `Expected QGIS 4.2.x`：先确认调用到的 QGIS 可执行文件和目标平台，不通过修改脚本绕过版本检查。
- profile 目录存在但插件菜单没有插件：核对 QGIS 实际打开的 profile、安装输出目录和 `deployment_manifest.json` 中的 `qgis_profile`；仍不一致时保留路径与 QGIS 版本信息交给管理员检查。
- PostgreSQL 连接失败：区分服务未监听、socket/host 不对、角色或数据库不存在、以及认证规则拒绝；不要把“安装了 PostgreSQL”当成连接已就绪。
- 连接成功但首次 Run 仍失败：记录实际 DSN（去除密码）、schema、QGIS profile、插件版本、运行项目路径和错误信息，再进入运行时诊断；连接检查本身不证明 Run 已可用。
