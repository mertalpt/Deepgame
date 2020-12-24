class DriveOperations:

  # Constants
  drive_path = '/content/gdrive'

  # Copy a shared Drive folder to own Drive account
  # Based on: https://stackoverflow.com/a/61052437
  def copy_shared_folder_to_drive(folder_id):
    # Authenticate to access your Drive
    from google.colab import auth
    auth.authenticate_user()

    # Get folder name
    from googleapiclient.discovery import build
    service = build('drive', 'v3')
    folder_name = service.files().get(fileId=folder_id).execute()['name']

    # Import a library and download the folder
    !wget -qnc https://github.com/segnolin/google-drive-folder-downloader/raw/master/download.py
    from download import download_folder
    download_folder(service, folder_id, './', folder_name)
    return folder_name


  # Copy from current notebook to Drive folder
  def copy_between(source_path, target_path, is_folder=True):
    import os, os.path

    # Mount Drive if not mounted
    if not os.path.isdir(DriveOps.drive_path):
      from google.colab import drive
      drive.mount(DriveOps.drive_path)

    if not os.path.exists(source_path):
      raise ValueError("Invalid source path!")

    # Copy files
    import shutil, errno
    try:
        shutil.copytree(source_path, target_path)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(source_path, target_path)
        else:
            raise
  
