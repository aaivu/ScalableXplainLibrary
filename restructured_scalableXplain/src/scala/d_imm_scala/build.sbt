name := "Spark-DIMM"

version := "1.0"

scalaVersion := "2.12.18"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core"  % "3.5.4" % "provided",
  "org.apache.spark" %% "spark-sql"   % "3.5.4" % "provided",
  "org.apache.spark" %% "spark-mllib" % "3.5.4" % "provided",
  "org.scalatest"    %% "scalatest"   % "3.2.16" % Test
)

enablePlugins(AssemblyPlugin)

Compile / mainClass := Some("dimm.driver.IMMWrapper")

assemblyJarName := "spark-dimm-assembly.jar"

import sbtassembly.AssemblyPlugin.autoImport._

assembly / assemblyMergeStrategy := {
  case PathList("google", "protobuf", xs @ _*) => MergeStrategy.first
  case PathList("META-INF", xs @ _*) =>
    xs.map(_.toLowerCase) match {
      case ("manifest.mf" :: Nil)     => MergeStrategy.discard
      case ("index.list" :: Nil)      => MergeStrategy.discard
      case ("dependencies" :: Nil)    => MergeStrategy.discard
      case ps if ps.exists(_.endsWith(".sf"))  => MergeStrategy.discard
      case ps if ps.exists(_.endsWith(".dsa")) => MergeStrategy.discard
      case _                          => MergeStrategy.first
    }
  case _ => MergeStrategy.first
}

// Exclude Scala and Spark jars from assembly JAR
assembly / fullClasspath := (Compile / fullClasspath).value filterNot { attr =>
  val name = attr.data.getName
  name.contains("spark-core") || name.contains("spark-sql") || name.contains("spark-mllib") || name.contains("scala-library")
}

Test / fork := true

Test / javaOptions ++= Seq(
  "--add-exports=java.base/sun.nio.ch=ALL-UNNAMED",
  "--add-opens=java.base/java.nio=ALL-UNNAMED"
)
