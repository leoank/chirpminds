{
  project.name = "pythonml";
  services = {
    mongo = {
      service = {
        image = "mongo";
        ports = [ "27017:27017" ];
        environment = {
          MONGO_INITDB_ROOT_USERNAME = "admin";
          MONGO_INITDB_ROOT_PASSWORD = "admin";
        };

      };
    };
    labelstudio = {
      service = {
        image = "heartexlabs/label-studio";
        environment = {
          LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true";
          LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = "/";
          DATA_UPLOAD_MAX_NUMBER_FILES = 1000000;
        };
        volumes = [
          "${toString ../../scratch/container_data/labelstudio}:/label-studio/data"
          "${toString ../../scratch/new_frames}:/datastore/frames"
        ];
        ports = [ "8080:8080" ];
        command = [
          "label-studio"
          "--host"
          "0.0.0.0"
          "--ml-backends"
          "http://10.13.84.1:9090"
        ];

      };
    };

    # testml = {
    #   out.service = {
    #     deploy.resources.reservations.devices = [
    #       {
    #         driver = "nvidia";
    #         capabilities = [ "gpu" ];
    #         count = 1;
    #       }
    #     ];
    #
    #   };
    #   service = {
    #     image = "ubuntu";
    #     # environment = {
    #     #   NVIDIA_VISIBLE_DEVICES = "all";
    #     #   NVIDIA_DRIVER_CAPABILITIES = "all";
    #     # };
    #     command = [ "nvidia-smi" ];
    #   };
    # };
    #
    # labelstudioml = {
    #   image.rawConfig = {
    #     deploy.resources.reservations.devices = [
    #       {
    #         driver = "nvidia";
    #         capabilities = [ "gpu" ];
    #         count = 1;
    #       }
    #     ];
    #   };
    #   service = {
    #     image = "heartexlabs/label-studio-ml-backend";
    #     ports = [ "9090:9090" ];
    #     environment = {
    #       LABEL_STUDIO_HOST = "http://labelstudio:8080";
    #       LABEL_STUDIO_ACCESS_TOKEN = "6b6776e528d90bfad8ee5d645066f2cebaa2f57d";
    #       SAM_CHOICE = "SAM";
    #     };
    #   };
    # };
  };

}
