---
title: Tomcat使用教程
date: 2025-04-30
categories: 
    - 学CS/SE
tags: 
    - 后端
desc: 软工基后端部署时候用到的工具，学习汇总。
---

参考：<a href="https://cloud.tencent.com/developer/article/1701463">https://cloud.tencent.com/developer/article/1701463</a>


## 简介

### 定义
Tomcat 是由 Apache 软件基金会开发的一款开源的 Java Servlet 容器。它实现了 Java Servlet 和 JavaServer Pages（JSP）技术，能够运行 Java Web 应用程序。Tomcat 通常用于开发和部署基于 Java 的 Web 应用，支持多种协议，包括 HTTP/1.1、HTTP/2（部分版本支持）等。许多企业级应用、中小型网站以及开发测试环境都广泛使用 Tomcat 作为 Web 服务器。

### 应用场景
  1. **开发测试环境** ：在 Java Web 开发过程中，Tomcat 是开发者最常用的测试服务器之一。它易于安装和配置，能够快速启动和停止，方便开发者在本地进行代码调试和功能测试。
  2. **中小型网站部署** ：对于一些访问量适中的中小型网站，Tomcat 可以直接作为生产环境的 Web 服务器。它提供了足够的性能和稳定性来满足这类网站的需求。
  3. **企业级应用** ：在企业级应用中，Tomcat 通常与负载均衡器、反向代理服务器等配合使用，以构建高可用、高性能的应用架构。例如，通过 Nginx 或 Apache 作为反向代理，将请求分发到多个 Tomcat 实例上，实现负载均衡和故障冗余。

<br>

## 安装

### 下载途径
  1. **官方网站下载** ：访问[Apache Tomcat 官方网站](https://tomcat.apache.org/)，在下载页面选择适合您操作系统的安装包。官方网站提供了不同版本的 Tomcat，包括稳定版和开发版。建议根据项目需求和系统环境选择合适的版本。例如，如果您的项目基于 Servlet 4.0 规范，且运行在 Java 8 环境下，Tomcat 9.0 是一个不错的选择。
  2. **第三方软件仓库** ：在一些 Linux 发行版的软件仓库中，也包含 Tomcat 安装包。例如，在 Ubuntu 系统中，可以使用命令`sudo apt-get install tomcat9`来安装 Tomcat 9.0。不过，通过这种方式安装的 Tomcat 可能不是最新版本，且配置方式可能与官方安装包有所不同。

### 安装步骤

#### Windows

1. **解压安装包** ：下载完成后，将安装包解压到您希望安装 Tomcat 的目录。例如，解压到`C:\Program Files\Apache\Tomcat9`。解压后，您将看到 Tomcat 的目录结构，包括`bin`、`conf`、`lib`、`webapps`等目录。

2. **设置环境变量（可选但推荐）** ：为了方便使用命令行操作 Tomcat，建议设置`CATALINA_HOME`环境变量。右键点击“此电脑”，选择“属性”，在“高级系统设置”中点击“环境变量”按钮。在“系统变量”区域，点击“新建”按钮，变量名输入`CATALINA_HOME`，变量值输入 Tomcat 的解压目录路径（如`C:\Program Files\Apache\Tomcat9`）。然后，在“系统变量”中的“Path”变量里，添加`%CATALINA_HOME%\bin`，这样您就可以在任意命令行窗口中使用 Tomcat 的命令（如`startup.bat`、`shutdown.bat`等）。

#### Linux

1. **解压安装包** ：使用命令`tar -zxvf apache-tomcat-9.x.x.tar.gz -C /usr/local/`（假设您希望将 Tomcat 安装在`/usr/local/`目录下）来解压安装包。解压后，Tomcat 将被放置在`/usr/local/apache-tomcat-9.x.x`目录中。

2. **设置环境变量（可选但推荐）** ：编辑`/etc/profile`文件，添加以下内容：

```bash
export CATALINA_HOME="/usr/local/apache-tomcat-9.x.x"
export PATH="$CATALINA_HOME/bin:$PATH"
```

然后，执行`source /etc/profile`命令使环境变量生效。这样，您就可以在任意终端窗口中使用 Tomcat 的命令行工具。




<br>

## 目录结构

1. **bin** ：存放 Tomcat 的启动、停止等命令脚本，如`startup.bat`、`shutdown.bat`（Windows）或`startup.sh`、`shutdown.sh`（Linux）。这些脚本用于控制 Tomcat 的运行状态。

2. **conf** ：存放 Tomcat 的配置文件，如`server.xml`、`web.xml`、`context.xml`等。

- `server.xml`：Tomcat 的核心配置文件，定义了服务、连接器、引擎、主机等组件的配置。例如，可以在这里配置 Tomcat 的监听端口、虚拟主机等。
- `web.xml`：Web 应用的默认配置文件，定义了 Servlet、过滤器、监听器等的配置。Tomcat 会为每个 Web 应用加载自己的`web.xml`文件，同时也会加载`conf/web.xml`作为全局默认配置。
- `context.xml`：定义了与 Web 应用相关的上下文配置，如数据源、资源引用等。可以在这里配置数据库连接池等资源。

3. **lib** ：存放 Tomcat 运行所需的库文件（JAR 包），如 Servlet API、JSP API 等。这些库文件会被 Tomcat 加载到类路径中，供 Web 应用使用。

4. **logs** ：存放 Tomcat 的日志文件，如`catalina.out`、`localhost.log`等。通过日志可以监控 Tomcat 的运行情况、排查问题等。
  
5. **webapps** ：存放部署的 Web 应用。每个 Web 应用都有自己的目录，目录名即为应用的上下文路径。例如，一个名为`myapp`的目录将对应上下文路径`/myapp`。当 Tomcat 启动时，会自动扫描该目录下的应用并进行部署。您也可以将 Web 应用打包成 WAR 文件，直接放置在`webapps`目录下，Tomcat 会自动解压并部署应用。

6. **work** ：Tomcat 在运行时会将 JSP 文件编译成 Servlet 类，存放在该目录下。通常情况下，开发人员不需要手动操作该目录，Tomcat 会自动管理其中的内容。

<br>

## 使用

### 基本操作命令

1. **启动 Tomcat** ：在 Windows 系统中，打开命令行窗口，进入`%CATALINA_HOME%\bin`目录（如`C:\Program Files\Apache\Tomcat9\bin`），执行`startup.bat`命令。在 Linux 系统中，进入`$CATALINA_HOME/bin`目录，执行`./startup.sh`命令。启动成功后，Tomcat 会在控制台输出启动日志，并监听配置的端口（默认为 8080）。
  
2. **停止 Tomcat** ：在 Windows 系统中，进入`%CATALINA_HOME%\bin`目录，执行`shutdown.bat`命令。在 Linux 系统中，进入`$CATALINA_HOME/bin`目录，执行`./shutdown.sh`命令。如果 Tomcat 没有及时停止，可以尝试使用任务管理器（Windows）或`kill`命令（Linux）来强制停止 Tomcat 进程。
  
3. **查看日志** ：Tomcat 的日志文件位于`%CATALINA_HOME%\logs`（Windows）或`$CATALINA_HOME/logs`（Linux）目录下。其中，`catalina.out`文件记录了 Tomcat 的标准输出和标准错误信息；`localhost.log`文件记录了 Web 应用的请求处理日志等。通过查看日志，可以了解 Tomcat 的运行状态、应用的错误信息等。



### 配置文件详解

（以 server.xml 为例）

`server.xml` 是 Tomcat 的核心配置文件，以下是一些常见的配置项及其说明：

1. **Service（服务）配置** ：

```xml
<Service name="Catalina">
    <!-- 其他配置 -->
</Service>
```

- `name`属性：服务的名称，通常为`Catalina`，表示这是一个标准的 Tomcat 服务。

2. **Connector（连接器）配置** ：

```xml
<Connector port="8080" protocol="HTTP/1.1"
           connectionTimeout="20000"
           redirectPort="8443"
           URIEncoding="UTF-8"/>
```

-  `port`属性：指定 Tomcat 监听的端口号，默认为 8080。您可以根据需要修改为其他可用端口，如 8081、80 等。
-  `protocol`属性：指定连接器使用的协议，如`HTTP/1.1`表示使用 HTTP 1.1 协议。Tomcat 还支持其他协议，如`org.apache.coyote.http11.Http11NioProtocol`（基于 NIO 的 HTTP 协议）等。
-  `connectionTimeout`属性：设置连接的超时时长，单位为毫秒。如果客户端在规定时间内没有完成请求，Tomcat 将关闭连接。
-  `redirectPort`属性：当客户端请求需要安全连接（HTTPS）时，Tomcat 将请求重定向到此端口。默认为 8443，可以配置自己的 HTTPS 连接器。
-  `URIEncoding`属性：指定请求 URI 的编码格式，设置为`UTF-8`可以解决请求参数乱码问题。

3. **Engine（引擎）配置** ：

```xml
<Engine name="Catalina" defaultHost="localhost">
    <!-- 其他配置 -->
</Engine>
```

-  `name`属性：引擎的名称，通常为`Catalina`。
-  `defaultHost`属性：指定默认的虚拟主机名称。当客户端请求的主机名无法匹配任何配置的主机时，将使用此默认主机处理请求。

4. **Host（主机）配置** ：

```xml
<Host name="localhost" appBase="webapps"
      unpackWARs="true" autoDeploy="true">
    <!-- 其他配置 -->
</Host>
```

- `name`属性：主机的名称，对于本地开发环境，通常为`localhost`。
- `appBase`属性：指定 Web 应用的基本目录，默认为`webapps`。Tomcat 会从该目录下加载和部署 Web 应用。
- `unpackWARs`属性：如果为`true`，Tomcat 会将 WAR 包解压后部署；如果为`false`，则直接从 WAR 包中运行应用（这可能会影响性能）。
- `autoDeploy`属性：如果为`true`，Tomcat 会自动检测`appBase`目录下的新应用或应用更新，并进行部署或重新部署。

<br>

### 部署后端例子
（以 Spring MVC Web 项目为例）

#### 项目准备

  1. **项目类型与结构** ：假设这是一个基于 Spring MVC 的 Web 项目，项目结构遵循标准的 Maven 项目结构。项目包含`src/main/java`（Java 源代码目录）、`src/main/resources`（资源配置文件目录）、`src/main/webapp`（Web 资源目录，如 JSP 文件、静态资源等）等目录。项目的打包类型为 WAR 包。

  2. **编译版本与依赖** ：项目使用 Java 8 作为编译版本，在`pom.xml`文件中指定`<maven.compiler.source>1.8</maven.compiler.source>`和`<maven.compiler.target>1.8</maven.compiler.target>`。依赖 Spring 5.3.24 版本，以及其他相关的依赖，如 Spring Web、Spring Context、JSTL 等。例如：

```xml
<dependencies>
    <!-- Spring Web 依赖 -->
    <dependency>
        <groupId>org.springframework</groupId>
        <artifactId>spring-webmvc</artifactId>
        <version>5.3.24</version>
    </dependency>
    <!-- Spring Context 依赖 -->
    <dependency>
        <groupId>org.springframework</groupId>
        <artifactId>spring-context</artifactId>
        <version>5.3.24</version>
    </dependency>
    <!-- JSTL 依赖 -->
    <dependency>
        <groupId>javax.servlet</groupId>
        <artifactId>javax.servlet-api</artifactId>
        <version>4.0.1</version>
        <scope>provided</scope>
    </dependency>
    <dependency>
        <groupId>javax.servlet.jsp</groupId>
        <artifactId>javax.servlet.jsp-api</artifactId>
        <version>2.3.3</version>
        <scope>provided</scope>
    </dependency>
    <dependency>
        <groupId>javax.servlet</groupId>
        <artifactId>jstl</artifactId>
        <version>1.2</version>
    </dependency>
</dependencies>
```

#### 项目配置

1. **web.xml 配置** ：在`src/main/webapp/WEB-INF/web.xml`文件中，配置 Spring MVC 的前端控制器`DispatcherServlet`，以及字符编码过滤器等。例如：

```xml
<web-app xmlns="http://xmlns.jcp.org/xml/ns/javaee"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://xmlns.jcp.org/xml/ns/javaee
         http://xmlns.jcp.org/xml/ns/javaee/web-app_4_0.xsd"
         version="4.0">

    <!-- 字符编码过滤器 -->
    <filter>
        <filter-name>characterEncodingFilter</filter-name>
        <filter-class>org.springframework.web.filter.CharacterEncodingFilter</filter-class>
        <init-param>
            <param-name>encoding</param-name>
            <param-value>UTF-8</param-value>
        </init-param>
        <init-param>
            <param-name>forceEncoding</param-name>
            <param-value>true</param-value>
        </init-param>
    </filter>
    <filter-mapping>
        <filter-name>characterEncodingFilter</filter-name>
        <url-pattern>/*</url-pattern>
    </filter-mapping>

    <!-- Spring MVC 前端控制器 -->
    <servlet>
        <servlet-name>dispatcherServlet</servlet-name>
        <servlet-class>org.springframework.web.servlet.DispatcherServlet</servlet-class>
        <init-param>
            <param-name>contextConfigLocation</param-name>
            <param-value>/WEB-INF/config/spring-mvc.xml</param-value>
        </init-param>
        <load-on-startup>1</load-on-startup>
    </servlet>
    <servlet-mapping>
        <servlet-name>dispatcherServlet</servlet-name>
        <url-pattern>/</url-pattern>
    </servlet-mapping>

    <!-- Spring 的 ContextLoaderListener -->
    <listener>
        <listener-class>org.springframework.web.context.ContextLoaderListener</listener-class>
    </listener>
    <context-param>
        <param-name>contextConfigLocation</param-name>
        <param-value>/WEB-INF/config/applicationContext.xml</param-value>
    </context-param>

</web-app>
```

- 字符编码过滤器：使用 Spring 提供的`CharacterEncodingFilter`，确保所有请求和响应都使用 UTF - 8 编码，避免乱码问题。

- `DispatcherServlet`：作为 Spring MVC 的前端控制器，处理所有进入应用的 HTTP 请求。通过`contextConfigLocation`参数指定 Spring MVC 的配置文件路径（`/WEB-INF/config/spring-mvc.xml`）。

- `ContextLoaderListener`：用于加载 Spring 的应用上下文，通过`contextConfigLocation`参数指定应用上下文配置文件路径（`/WEB-INF/config/applicationContext.xml`）。

2. **数据库连接配置（applicationContext.xml）** ：在`src/main/webapp/WEB-INF/config/applicationContext.xml`文件中，配置数据源等信息。例如：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xmlns:context="http://www.springframework.org/schema/context"
       xsi:schemaLocation="http://www.springframework.org/schema/beans
           http://www.springframework.org/schema/beans/spring-beans.xsd
           http://www.springframework.org/schema/context
           http://www.springframework.org/schema/context/spring-context.xsd">

    <!-- 数据源配置 -->
    <bean id="dataSource" class="org.springframework.jdbc.datasource.DriverManagerDataSource">
        <property name="driverClassName" value="com.mysql.cj.jdbc.Driver"/>
        <property name="url" value="jdbc:mysql://localhost:3306/mydatabase?useUnicode=true&characterEncoding=UTF-8&serverTimezone=UTC"/>
        <property name="username" value="root"/>
        <property name="password" value="password"/>
    </bean>

    <!-- 其他配置（如事务管理器、Spring 组件扫描等） -->
    <bean id="transactionManager" class="org.springframework.jdbc.datasource.DataSourceTransactionManager">
        <property name="dataSource" ref="dataSource"/>
    </bean>

    <context:component-scan base-package="com.example"/>

</beans>
```

- 数据源配置：使用`DriverManagerDataSource`配置数据库连接信息，包括数据库驱动类名、数据库 URL、用户名和密码。确保数据库连接配置正确，否则应用无法正常连接到数据库。

- 组件扫描：通过`<context:component-scan>`标签指定 Spring 组件扫描的包路径（如`com.example`），使得 Spring 能够自动检测并管理标注了`@Component`、`@Service`、`@Repository`等注解的类。

#### 编译打包

1. **使用 Maven 编译打包** ：在项目根目录下（包含`pom.xml`文件的目录），打开命令行窗口，执行`mvn clean package`命令。Maven 会根据`pom.xml`文件中的配置，先清理项目（删除`target`目录下的旧文件），然后编译项目源代码、运行测试用例，最后将项目打包成 WAR 文件。打包完成后，WAR 文件将生成在`target`目录下，文件名为`course_selection_backend.war`（假设项目最终打包文件名为`course_selection_backend.war`）。

2. **验证打包结果** ：进入`target`目录，确保`course_selection_backend.war`文件存在且大小符合预期。如果打包过程中出现错误，根据错误提示信息修复相应的问题，例如依赖缺失、代码编译错误等。

#### 部署到 Tomcat

1. **复制 WAR 文件** ：找到编译生成的`course_selection_backend.war`文件（如`D:\User\Desktop\course_selection_backend\target\course_selection_backend.war`），将其复制到 Tomcat 的`webapps`目录下（如`C:\Program Files\Apache\Tomcat9\webapps`或`/usr/local/apache-tomcat-9.x.x/webapps`）。Tomcat 会自动检测到新的 WAR 文件，并在后台进行解压和部署。如果 Tomcat 正在运行，您可以在 Tomcat 的`logs/catalina.out`文件中看到部署日志，了解部署进度。

2. **启动 Tomcat（如果尚未启动）** ：如果 Tomcat 尚未启动，进入`%CATALINA_HOME%\bin`（Windows）或`$CATALINA_HOME/bin`（Linux）目录，执行`startup.bat`（Windows）或`./startup.sh`（Linux）命令启动 Tomcat。启动后，Tomcat 会继续完成对新 WAR 文件的部署。

3. **验证部署** ：打开浏览器，访问`http://localhost:8080/course_selection_backend/`（假设应用的上下文路径为`course_selection_backend`）。如果页面能够正常显示，或者返回预期的 API 响应，说明部署成功。例如，如果项目中有一个首页控制器映射到`/`路径，访问该地址应显示首页内容。

<br>

## 高级配置与优化

### 性能优化

1. **调整线程池配置** ：在`server.xml`文件中，可以对 Tomcat 的线程池进行配置，以提高并发处理能力。例如：

```xml
<Executor name="tomcatThreadPool" namePrefix="catalina-exec-"
          maxThreads="200" minSpareThreads="50"/>
```

- 创建一个名为`tomcatThreadPool`的执行器，设置最大线程数为 200，最小空闲线程数为 50。然后，在`Connector`标签中引用该执行器：

```xml
<Connector executor="tomcatThreadPool"
           port="8080" protocol="HTTP/1.1"
           connectionTimeout="20000"
           redirectPort="8443"
           URIEncoding="UTF-8"/>
```

通过调整线程池参数，可以根据服务器的硬件资源和应用的负载情况，优化 Tomcat 的并发性能。

2. **启用压缩** ：在`Connector`标签中添加`compression`、`compressableMimeType`等属性，启用响应内容压缩，减少网络传输数据量。例如：

```xml
<Connector port="8080" protocol="HTTP/1.1"
           connectionTimeout="20000"
           redirectPort="8443"
           URIEncoding="UTF-8"
           compression="on"
           compressableMimeType="text/html,text/xml,text/plain,application/json,application/javascript"/>
```

- `compression="on"`表示启用压缩功能。
- `compressableMimeType`属性指定哪些 MIME 类型的内容可以被压缩，如`text/html`、
- `application/json`等。对于文本类内容，压缩可以显著减少传输大小，提高响应速度。

3. **优化内存配置** ：通过设置 JVM 参数，调整 Tomcat 的内存分配。在启动 Tomcat 的脚本（如`catalina.sh`或`catalina.bat`）中，修改`JAVA_OPTS`变量，增加内存配置。例如：

```bash
export JAVA_OPTS="-Xms2048m -Xmx2048m -XX:MaxPermSize=512m"
```

-（对于 Linux 系统）设置初始内存为 2048MB，最大内存为 2048MB，永久代大小为 512MB（对于 Java 8 及以下版本）。适当的内存配置可以避免内存溢出问题，提高 Tomcat 的稳定性和性能。不过，内存参数的调整需要根据服务器的实际物理内存和应用的需求进行合理设置。

### 安全配置

1. **配置 HTTPS** ：为了保护数据传输安全，可以为 Tomcat 配置 HTTPS 连接。首先，需要生成一个密钥库（keystore）文件，可以使用 Java 自带的`keytool`命令生成。例如：

```bash
keytool -genkeypair -alias tomcat -keyalg RSA -keysize 2048 -keystore /path/to/keystore.jks -storepass mypassword -keypass mypassword -validity 365 -dname "CN=localhost, OU=MyUnit, O=MyOrg, L=MyCity, ST=MyState, C=US"
```

- 生成一个名为`keystore.jks`的密钥库文件，密码为`mypassword`，有效期为 365 天。然后，在`server.xml`文件中配置 HTTPS 连接器：

```xml
<Connector port="8443" protocol="org.apache.coyote.http11.Http11NioProtocol"
           maxThreads="150" SSLEnabled="true">
    <SSLHostConfig>
        <Certificate certificateKeystoreFile="/path/to/keystore.jks"
                     type="RSA" certificateKeystorePassword="mypassword"/>
    </SSLHostConfig>
</Connector>
```

- 配置完成后，重启 Tomcat，即可通过`https://localhost:8443`访问应用。为了确保安全性，建议在生产环境中使用由权威证书颁发机构（CA）签发的证书，而不是自签名证书。

2. **限制访问权限** ：通过配置`web.xml`文件中的安全约束，可以限制对某些资源的访问权限。例如：

```xml
<security-constraint>
    <web-resource-collection>
        <web-resource-name>Protected Area</web-resource-name>
        <url-pattern>/admin/*</url-pattern>
    </web-resource-collection>
    <auth-constraint>
        <role-name>admin</role-name>
    </auth-constraint>
</security-constraint>
```

- 此配置表示对`/admin/*`路径下的资源进行保护，只有具有`admin`角色的用户才能访问。同时，需要配置登录配置：

```xml
<login-config>
    <auth-method>BASIC</auth-method>
    <realm-name>Admin Realm</realm-name>
</login-config>
```

- 指定使用基本认证方式（BASIC），并指定领域名为`Admin Realm`。然后，在 Tomcat 的`conf/tomcat-users.xml`文件中配置用户和角色：

```xml
<tomcat-users>
    <role rolename="admin"/>
    <user username="admin" password="adminpassword" roles="admin"/>
</tomcat-users>
```

- 通过这些配置，可以限制对敏感资源的访问，提高应用的安全性。

### （集群与负载均衡配置

1. **Tomcat 集群配置** ：为了实现高可用性和负载均衡，可以配置多个 Tomcat 实例组成集群。在每个 Tomcat 实例的`server.xml`文件中，添加集群相关配置。例如：

```xml
<Cluster className="org.apache.catalina.ha.tcp.SimpleTcpCluster">
    <Manager className="org.apache.catalina.ha.session.DeltaManager"
             expireSessionsOnShutdown="false"
             notifyListenersOnReplication="true"/>
    <Channel className="org.apache.catalina.tribes.group.GroupChannel">
        <Membership className="org.apache.catalina.tribes.membership.McastService"
                    address="228.0.0.4"
                    port="45564"
                    frequency="500"
                    dropTime="3000"/>
        <Receiver className="org.apache.catalina.tribes.transport.nio.NioReceiver"
                  address="auto"
                  port="4000"
                  autoBind="100"
                  selectorTimeout="5000"
                  maxThreads="6"/>
        <Sender className="org.apache.catalina.tribes.transport.ReplicationTransmitter">
            <Transport className="org.apache.catalina.tribes.transport.nio.PooledParallelSender"/>
        </Sender>
    </Channel>
    <Valve className="org.apache.catalina.ha.tcp.ReplicationValve"
           filter=".*\.jpg|.*\.gif|.*\.js|.*\.css|.*\.png|.*\.js"/>
    <Deployer className="org.apache.catalina.ha.deploy.FarmWarDeployer"
              tempDir="${catalina.base}/temp"/>
    <ClusterListener className="org.apache.catalina.ha.session.ClusterSessionListener"/>
</Cluster>
```

- 配置集群管理器、通信通道（包括成员广播、接收器、发送器等）以及复制阀等组件。每个 Tomcat 实例使用相同的集群配置（除了接收器的端口号等可能不同），通过组播（Membership 配置中的`address`和`port`）发现集群中的其他成员，并进行会话同步和请求分发。

2. **负载均衡配置（使用 Nginx 作为反向代理）** ：在 Nginx 配置文件中，定义一个 upstream 服务器组，包含多个 Tomcat 实例的地址和端口。例如：

```nginx
upstream tomcat_cluster {
    server 192.168.1.101:8080;
    server 192.168.1.102:8080;
    server 192.168.1.103:8080;
}
```

- 然后，在 server 配置块中，将请求代理到 upstream 服务器组：

```nginx
server {
    listen 80;
    server_name www.example.com;

    location / {
        proxy_pass http://tomcat_cluster;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Connection keep-alive;
    }
}
```

- Nginx 会根据默认的轮询算法将请求分发到不同的 Tomcat 实例上，实现负载均衡。您还可以根据需要配置其他负载均衡算法，如最少连接数、IP 哈希等。

<br>

## 监控与维护

### 监控工具与指标

1. **JMX（Java Management Extensions）监控** ：Tomcat 提供了 JMX 支持，可以通过 JConsole、VisualVM 等工具连接到 Tomcat 进程，监控其运行时信息。主要监控指标包括：

- **线程池指标** ：线程数（currentThreadCount）、最大线程数（maxThreadCount）、忙线程数（currentThreadsBusy）等。通过这些指标，可以了解 Tomcat 的并发处理能力是否充足，是否存在线程瓶颈。

- **连接数指标** ：最大连接数（maxConnections）、已连接数（connectionCount）等。用于监控 Tomcat 的连接使用情况，防止连接数过多导致资源耗尽。

- **请求处理指标** ：处理请求总数（requestCount）、错误请求数（errorCount）、处理时间（processingTime）等。这些指标可以帮助评估应用的性能和稳定性，发现潜在的性能问题。

2. **Tomcat Manager App** ：Tomcat 自带了一个管理应用（Manager App），可以通过浏览器访问`http://localhost:8080/manager/html`（需要配置管理员用户）。在`conf/tomcat-users.xml`文件中添加管理员用户：

```xml
<tomcat-users>
    <role rolename="manager-gui"/>
    <user username="admin" password="adminpassword" roles="manager-gui"/>
</tomcat-users>
```

- 登录后，可以查看 Tomcat 的运行状态、部署的应用、服务器信息等，还可以进行应用的部署、卸载、重新部署等操作。

### 日常维护任务

1. **定期备份** ：定期备份 Tomcat 的配置文件（如`server.xml`、`web.xml`等）、应用的 WAR 文件或目录、数据库连接配置等重要信息。备份可以存储在本地或远程存储设备上，以便在出现故障时能够快速恢复。

2. **日志分析** ：定期查看 Tomcat 的日志文件（如`catalina.out`、`localhost.log`等），分析其中的错误信息、警告信息等。通过日志分析，可以及时发现应用的异常情况、性能问题、安全漏洞等，并采取相应的措施进行修复。
  
3. **更新与补丁管理** ：关注 Tomcat 官方发布的更新和安全补丁，及时升级到最新版本。同时，更新相关的依赖库（如 Servlet 容器、连接池等）和操作系统补丁，以确保系统的安全性和稳定性。

<br>

## 常见问题与解决方案

### 启动失败

  1. **端口被占用** ：如果 Tomcat 无法启动，并提示“Address already in use”错误，可能是指定的端口（如 8080）被其他进程占用。可以通过命令行工具（如 Windows 的`netstat -aon`或 Linux 的`netstat -tulnp`）查找占用该端口的进程，并结束该进程。或者，修改`server.xml`文件中的端口号，使用其他可用端口。
  2. **配置文件错误** ：Tomcat 的配置文件（如`server.xml`）如果存在语法错误或不正确的配置，可能导致启动失败。检查配置文件的 XML 格式是否正确，各标签是否闭合，属性值是否合法。可以尝试使用 XML 验证工具对配置文件进行验证。
  3. **内存不足** ：如果服务器内存资源不足，Tomcat 可能在启动过程中因内存分配失败而崩溃。增加服务器的物理内存或调整 Tomcat 的 JVM 内存参数（如`-Xms`、`-Xmx`），确保 Tomcat 有足够的内存空间运行。

### 乱码修复

原因：
1. **文件编码问题** ：在开发过程中，如果 Java 源文件、JSP 文件等不是以 UTF - 8 编码保存，可能导致中文字符在传输过程中出现乱码。
  
2. **请求参数编码问题** ：当客户端（如浏览器）向服务器发送请求时，如果请求参数的编码格式与服务器预期的编码格式不一致，也会导致乱码。例如，客户端以`GBK`编码发送参数，而服务器以`UTF-8`解码，就会出现乱码。

3. **响应内容编码问题** ：服务器在向客户端发送响应时，如果没有正确设置响应内容的编码格式，客户端可能无法正确解析响应内容中的中文字符。

修复方法:
找到 Tomcat 目录下 `conf` 文件夹中的 `logging.properties` 文件，打开该文件，找到`java.util.logging.ConsoleHandler.encoding`, 修改编码为`GBK`。

### 应用部署失败

  1. **WAR 文件损坏或不完整** ：如果部署 WAR 文件时，Tomcat 提示“Failed to deploy”等错误，可能是 WAR 文件在打包或传输过程中损坏或不完整。重新编译打包项目，确保生成的 WAR 文件完整无误，然后重新部署。
  2. **应用配置错误** ：应用的配置文件（如`web.xml`、Spring 配置文件等）如果存在错误，可能导致部署失败。检查配置文件中的语法错误、类路径问题、资源引用问题等。例如，Spring 配置文件中引用了不存在的 Bean，或者`web.xml`中配置了错误的 Servlet 映射。
  3. **依赖冲突** ：如果应用使用了多个版本的相同依赖库（如不同版本的 Spring 框架），可能导致类加载冲突，从而部署失败。检查项目的依赖配置（如`pom.xml`），排除依赖冲突，确保使用的依赖库版本兼容。

### 性能问题

  1. **响应时间过长** ：如果应用的响应时间过长，可能是由于代码性能瓶颈、数据库查询效率低、Tomcat 线程池配置不合理等原因导致。通过性能分析工具（如 VisualVM、Arthas 等）分析应用的运行时性能，定位耗时的方法或代码块。优化数据库查询语句，添加索引，减少不必要的计算和 I/O 操作。调整 Tomcat 的线程池参数，增加线程数等，以提高并发处理能力。
  2. **内存泄漏** ：如果 Tomcat 运行一段时间后，出现内存溢出（`OutOfMemoryError`）错误，可能是应用存在内存泄漏问题。使用内存分析工具（如 Eclipse Memory Analyzer）分析 Tomcat 的堆转储（heap dump）文件，查找内存泄漏的根源，如未正确关闭的资源（数据库连接、文件流等）、静态集合中不断累积的对象等，并进行修复。

