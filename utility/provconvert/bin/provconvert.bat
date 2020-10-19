@REM ----------------------------------------------------------------------------
@REM  Copyright 2001-2006 The Apache Software Foundation.
@REM
@REM  Licensed under the Apache License, Version 2.0 (the "License");
@REM  you may not use this file except in compliance with the License.
@REM  You may obtain a copy of the License at
@REM
@REM       http://www.apache.org/licenses/LICENSE-2.0
@REM
@REM  Unless required by applicable law or agreed to in writing, software
@REM  distributed under the License is distributed on an "AS IS" BASIS,
@REM  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@REM  See the License for the specific language governing permissions and
@REM  limitations under the License.
@REM ----------------------------------------------------------------------------
@REM
@REM   Copyright (c) 2001-2006 The Apache Software Foundation.  All rights
@REM   reserved.

@echo off

set ERROR_CODE=0

:init
@REM Decide how to startup depending on the version of windows

@REM -- Win98ME
if NOT "%OS%"=="Windows_NT" goto Win9xArg

@REM set local scope for the variables with windows NT shell
if "%OS%"=="Windows_NT" @setlocal

@REM -- 4NT shell
if "%eval[2+2]" == "4" goto 4NTArgs

@REM -- Regular WinNT shell
set CMD_LINE_ARGS=%*
goto WinNTGetScriptDir

@REM The 4NT Shell from jp software
:4NTArgs
set CMD_LINE_ARGS=%$
goto WinNTGetScriptDir

:Win9xArg
@REM Slurp the command line arguments.  This loop allows for an unlimited number
@REM of arguments (up to the command line limit, anyway).
set CMD_LINE_ARGS=
:Win9xApp
if %1a==a goto Win9xGetScriptDir
set CMD_LINE_ARGS=%CMD_LINE_ARGS% %1
shift
goto Win9xApp

:Win9xGetScriptDir
set SAVEDIR=%CD%
%0\
cd %0\..\.. 
set BASEDIR=%CD%
cd %SAVEDIR%
set SAVE_DIR=
goto repoSetup

:WinNTGetScriptDir
for %%i in ("%~dp0..") do set "BASEDIR=%%~fi"

:repoSetup
set REPO=


if "%JAVACMD%"=="" set JAVACMD=java

if "%REPO%"=="" set REPO=%BASEDIR%\repo

set CLASSPATH="%BASEDIR%"\etc;"%REPO%"\org\openprovenance\prov\prov-xml\0.9.2\prov-xml-0.9.2.jar;"%REPO%"\org\openprovenance\prov\prov-model\0.9.2\prov-model-0.9.2.jar;"%REPO%"\org\apache\commons\commons-lang3\3.9\commons-lang3-3.9.jar;"%REPO%"\javax\xml\bind\jaxb-api\2.3.1\jaxb-api-2.3.1.jar;"%REPO%"\org\apache\commons\commons-collections4\4.4\commons-collections4-4.4.jar;"%REPO%"\commons-io\commons-io\2.6\commons-io-2.6.jar;"%REPO%"\org\glassfish\jaxb\jaxb-runtime\2.3.1\jaxb-runtime-2.3.1.jar;"%REPO%"\org\glassfish\jaxb\txw2\2.3.1\txw2-2.3.1.jar;"%REPO%"\com\sun\istack\istack-commons-runtime\3.0.7\istack-commons-runtime-3.0.7.jar;"%REPO%"\org\jvnet\staxex\stax-ex\1.8\stax-ex-1.8.jar;"%REPO%"\com\sun\xml\fastinfoset\FastInfoset\1.2.15\FastInfoset-1.2.15.jar;"%REPO%"\javax\activation\javax.activation-api\1.2.0\javax.activation-api-1.2.0.jar;"%REPO%"\xerces\xercesImpl\2.12.0\xercesImpl-2.12.0.jar;"%REPO%"\xml-apis\xml-apis\1.4.01\xml-apis-1.4.01.jar;"%REPO%"\org\openprovenance\prov\prov-n\0.9.2\prov-n-0.9.2.jar;"%REPO%"\org\antlr\antlr-runtime\3.4\antlr-runtime-3.4.jar;"%REPO%"\antlr\antlr\2.7.7\antlr-2.7.7.jar;"%REPO%"\org\antlr\stringtemplate\4.0.2\stringtemplate-4.0.2.jar;"%REPO%"\org\openprovenance\prov\prov-rdf\0.9.2\prov-rdf-0.9.2.jar;"%REPO%"\org\openrdf\sesame\sesame-runtime\4.1.2\sesame-runtime-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-model\4.1.2\sesame-model-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-repository-api\4.1.2\sesame-repository-api-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-query\4.1.2\sesame-query-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-repository-manager\4.1.2\sesame-repository-manager-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-repository-event\4.1.2\sesame-repository-event-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-repository-http\4.1.2\sesame-repository-http-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-repository-sparql\4.1.2\sesame-repository-sparql-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-repository-contextaware\4.1.2\sesame-repository-contextaware-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-repository-sail\4.1.2\sesame-repository-sail-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-queryalgebra-model\4.1.2\sesame-queryalgebra-model-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-queryalgebra-evaluation\4.1.2\sesame-queryalgebra-evaluation-4.1.2.jar;"%REPO%"\org\mapdb\mapdb\1.0.7\mapdb-1.0.7.jar;"%REPO%"\org\openrdf\sesame\sesame-http-client\4.1.2\sesame-http-client-4.1.2.jar;"%REPO%"\org\apache\httpcomponents\httpclient\4.5.2\httpclient-4.5.2.jar;"%REPO%"\org\apache\httpcomponents\httpcore\4.4.4\httpcore-4.4.4.jar;"%REPO%"\commons-codec\commons-codec\1.10\commons-codec-1.10.jar;"%REPO%"\org\openrdf\sesame\sesame-sail-api\4.1.2\sesame-sail-api-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-sail-federation\4.1.2\sesame-sail-federation-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-queryparser-api\4.1.2\sesame-queryparser-api-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-queryparser-serql\4.1.2\sesame-queryparser-serql-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-queryparser-sparql\4.1.2\sesame-queryparser-sparql-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-queryresultio-api\4.1.2\sesame-queryresultio-api-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-queryresultio-binary\4.1.2\sesame-queryresultio-binary-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-queryresultio-sparqljson\4.1.2\sesame-queryresultio-sparqljson-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-queryresultio-sparqlxml\4.1.2\sesame-queryresultio-sparqlxml-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-queryresultio-text\4.1.2\sesame-queryresultio-text-4.1.2.jar;"%REPO%"\com\opencsv\opencsv\3.2\opencsv-3.2.jar;"%REPO%"\org\openrdf\sesame\sesame-repository-dataset\4.1.2\sesame-repository-dataset-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-http-protocol\4.1.2\sesame-http-protocol-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-rio-api\4.1.2\sesame-rio-api-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-rio-datatypes\4.1.2\sesame-rio-datatypes-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-rio-languages\4.1.2\sesame-rio-languages-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-rio-binary\4.1.2\sesame-rio-binary-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-rio-jsonld\4.1.2\sesame-rio-jsonld-4.1.2.jar;"%REPO%"\com\github\jsonld-java\jsonld-java\0.7.0\jsonld-java-0.7.0.jar;"%REPO%"\org\apache\httpcomponents\httpclient-osgi\4.2.5\httpclient-osgi-4.2.5.jar;"%REPO%"\org\apache\httpcomponents\httpmime\4.2.5\httpmime-4.2.5.jar;"%REPO%"\org\apache\httpcomponents\httpclient-cache\4.2.5\httpclient-cache-4.2.5.jar;"%REPO%"\org\apache\httpcomponents\fluent-hc\4.2.5\fluent-hc-4.2.5.jar;"%REPO%"\org\apache\httpcomponents\httpcore-osgi\4.2.5\httpcore-osgi-4.2.5.jar;"%REPO%"\org\apache\httpcomponents\httpcore-nio\4.2.5\httpcore-nio-4.2.5.jar;"%REPO%"\org\slf4j\jcl-over-slf4j\1.7.9\jcl-over-slf4j-1.7.9.jar;"%REPO%"\org\openrdf\sesame\sesame-rio-ntriples\4.1.2\sesame-rio-ntriples-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-rio-nquads\4.1.2\sesame-rio-nquads-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-rio-rdfjson\4.1.2\sesame-rio-rdfjson-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-rio-trix\4.1.2\sesame-rio-trix-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-rio-turtle\4.1.2\sesame-rio-turtle-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-sail-inferencer\4.1.2\sesame-sail-inferencer-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-sail-model\4.1.2\sesame-sail-model-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-sail-lucene\4.1.2\sesame-sail-lucene-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-sail-lucene-api\4.1.2\sesame-sail-lucene-api-4.1.2.jar;"%REPO%"\com\spatial4j\spatial4j\0.4.1\spatial4j-0.4.1.jar;"%REPO%"\org\apache\lucene\lucene-core\5.1.0\lucene-core-5.1.0.jar;"%REPO%"\org\apache\lucene\lucene-queries\5.1.0\lucene-queries-5.1.0.jar;"%REPO%"\org\apache\lucene\lucene-highlighter\5.1.0\lucene-highlighter-5.1.0.jar;"%REPO%"\org\apache\lucene\lucene-join\5.1.0\lucene-join-5.1.0.jar;"%REPO%"\org\apache\lucene\lucene-memory\5.1.0\lucene-memory-5.1.0.jar;"%REPO%"\org\apache\lucene\lucene-analyzers-common\5.1.0\lucene-analyzers-common-5.1.0.jar;"%REPO%"\org\apache\lucene\lucene-queryparser\5.1.0\lucene-queryparser-5.1.0.jar;"%REPO%"\org\apache\lucene\lucene-sandbox\5.1.0\lucene-sandbox-5.1.0.jar;"%REPO%"\org\apache\lucene\lucene-spatial\5.1.0\lucene-spatial-5.1.0.jar;"%REPO%"\org\apache\lucene\lucene-misc\5.1.0\lucene-misc-5.1.0.jar;"%REPO%"\org\codehaus\mojo\animal-sniffer-annotations\1.14\animal-sniffer-annotations-1.14.jar;"%REPO%"\org\openrdf\sesame\sesame-sail-memory\4.1.2\sesame-sail-memory-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-sail-base\4.1.2\sesame-sail-base-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-sail-spin\4.1.2\sesame-sail-spin-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-spin\4.1.2\sesame-spin-4.1.2.jar;"%REPO%"\commons-lang\commons-lang\2.6\commons-lang-2.6.jar;"%REPO%"\org\openrdf\sesame\sesame-queryrender\4.1.2\sesame-queryrender-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-sail-nativerdf\4.1.2\sesame-sail-nativerdf-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-rio-n3\4.1.2\sesame-rio-n3-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-rio-rdfxml\4.1.2\sesame-rio-rdfxml-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-util\4.1.2\sesame-util-4.1.2.jar;"%REPO%"\org\openrdf\sesame\sesame-rio-trig\4.1.2\sesame-rio-trig-4.1.2.jar;"%REPO%"\org\openprovenance\prov\prov-interop\0.9.2\prov-interop-0.9.2.jar;"%REPO%"\commons-cli\commons-cli\1.4\commons-cli-1.4.jar;"%REPO%"\org\jboss\spec\javax\ws\rs\jboss-jaxrs-api_2.1_spec\2.0.0.Final\jboss-jaxrs-api_2.1_spec-2.0.0.Final.jar;"%REPO%"\org\openprovenance\prov\prov-json\0.9.2\prov-json-0.9.2.jar;"%REPO%"\com\google\code\gson\gson\2.8.5\gson-2.8.5.jar;"%REPO%"\org\openprovenance\prov\prov-dot\0.9.2\prov-dot-0.9.2.jar;"%REPO%"\org\openprovenance\prov\prov-template\0.9.2\prov-template-0.9.2.jar;"%REPO%"\com\fasterxml\jackson\core\jackson-annotations\2.9.9\jackson-annotations-2.9.9.jar;"%REPO%"\com\fasterxml\jackson\core\jackson-databind\2.9.9\jackson-databind-2.9.9.jar;"%REPO%"\com\fasterxml\jackson\core\jackson-core\2.9.9\jackson-core-2.9.9.jar;"%REPO%"\org\openprovenance\prov\prov-template-compiler\0.9.2\prov-template-compiler-0.9.2.jar;"%REPO%"\com\squareup\javapoet\1.11.1\javapoet-1.11.1.jar;"%REPO%"\com\google\guava\guava\19.0\guava-19.0.jar;"%REPO%"\org\apache\maven\maven-model\3.6.1\maven-model-3.6.1.jar;"%REPO%"\org\codehaus\plexus\plexus-utils\3.2.0\plexus-utils-3.2.0.jar;"%REPO%"\org\openprovenance\prov\prov-generator\0.9.2\prov-generator-0.9.2.jar;"%REPO%"\org\slf4j\slf4j-nop\1.7.26\slf4j-nop-1.7.26.jar;"%REPO%"\org\slf4j\slf4j-api\1.7.26\slf4j-api-1.7.26.jar;"%REPO%"\log4j\log4j\1.2.17\log4j-1.2.17.jar;"%REPO%"\org\openprovenance\prov\provconvert\0.9.2\provconvert-0.9.2.jar

set ENDORSED_DIR=
if NOT "%ENDORSED_DIR%" == "" set CLASSPATH="%BASEDIR%"\%ENDORSED_DIR%\*;%CLASSPATH%

if NOT "%CLASSPATH_PREFIX%" == "" set CLASSPATH=%CLASSPATH_PREFIX%;%CLASSPATH%

@REM Reaching here means variables are defined and arguments have been captured
:endInit

%JAVACMD% %JAVA_OPTS%  -classpath %CLASSPATH% -Dapp.name="provconvert" -Dapp.repo="%REPO%" -Dapp.home="%BASEDIR%" -Dbasedir="%BASEDIR%" org.openprovenance.prov.interop.CommandLineArguments %CMD_LINE_ARGS%
if %ERRORLEVEL% NEQ 0 goto error
goto end

:error
if "%OS%"=="Windows_NT" @endlocal
set ERROR_CODE=%ERRORLEVEL%

:end
@REM set local scope for the variables with windows NT shell
if "%OS%"=="Windows_NT" goto endNT

@REM For old DOS remove the set variables from ENV - we assume they were not set
@REM before we started - at least we don't leave any baggage around
set CMD_LINE_ARGS=
goto postExec

:endNT
@REM If error code is set to 1 then the endlocal was done already in :error.
if %ERROR_CODE% EQU 0 @endlocal


:postExec

if "%FORCE_EXIT_ON_ERROR%" == "on" (
  if %ERROR_CODE% NEQ 0 exit %ERROR_CODE%
)

exit /B %ERROR_CODE%
