pipeline {
  agent any
  options {
    disableConcurrentBuilds()
  }
  environment {
    registry = "rma-tools-docker-local.repo.vito.be"
    registryCredential = "svc_git_rma"
    repository = "guppy2_server"
    version = get_version()
    datetime = get_datetime()
    deploy_project = "guppy2_deploy"
    deploy_test_branch = "test"
    deploy_prod_branch = "prod"
  }
  stages {
    stage("Build conda-env docker image") {
      // build and name the conda-env stage of the Dockerfile to prevent dangling images
      steps {
        script {
          docker.build(repository + "_conda-env" + ":latest-${env.BRANCH_NAME}", "--target conda-env server")
        }
      }
    }
    stage("Build docker image") {
      steps {
        script {
          dockerImage = docker.build(repository + ":latest-${env.BRANCH_NAME}", "server")
        }
      }
    }
    stage("Run unittests") {
      steps {
        script {
          dockerImage.inside() {
            sh '/opt/guppy2/server/conda-env/bin/python -m unittest discover -s server -v'
          }
        }
      }
    }
    stage("Push docker image to registry") {
      steps {
        script {
          docker.withRegistry("https://" + registry, registryCredential ) {
            dockerImage.push("$version")
            if (env.BRANCH_NAME =~ /^(main|master)$/ ) {
              dockerImage.push("latest")
              dockerImage.push("$datetime")
            }
            if (env.TAG_NAME) {
              dockerImage.push("${env.TAG_NAME}")
            } else {
              dockerImage.push("latest-${env.BRANCH_NAME}")
            }
          }
          // Cleanup
          sh(script: "docker image rm $registry/$repository:$version")
          if (env.BRANCH_NAME =~ /^(main|master)$/ ) {
            sh(script: "docker image rm $registry/$repository:latest")
            sh(script: "docker image rm $registry/$repository:$datetime")
          }
          if (env.TAG_NAME) {
            sh(script: "docker image rm $registry/$repository:${env.TAG_NAME}")
          } else {
            sh(script: "docker image rm $registry/$repository:latest-${env.BRANCH_NAME}")
          }
        }
      }
    }
    stage("Update services") {
      when {
        expression { BRANCH_NAME =~ /^(develop|main|master)$/ }
      }
      steps {
        script {
          if (env.BRANCH_NAME == 'develop') {
            env.deploy_branch = deploy_test_branch
          }
          if (env.BRANCH_NAME =~ /^(main|master)$/ ) {
            env.deploy_branch = deploy_prod_branch
          }
          try {
            build(job: "${deploy_project}/${deploy_branch}", wait: true)
          } catch (hudson.AbortException e) {
            println("Skipping service update: ${deploy_project}/${deploy_branch} does not exist yet!")
          }
        }
      }
    }
  }
  post {
    always {
      deleteDir()
    }
  }
}

def get_version() {
  if (env.TAG_NAME) {
    return get_datetime() + "-${env.TAG_NAME}"
  }
  return get_datetime() + "-${env.BRANCH_NAME}"
}

def get_datetime() {
  return sh(script: "date -d @`git log -1 --format=%at` +%Y%m%d%Z%H%M", returnStdout: true).trim()
}
