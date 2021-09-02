import cv2


def load_cv2file(cv2file_path, nodes_list):
    cv_file = cv2.FileStorage(cv2file_path, cv2.FILE_STORAGE_READ)
    coefficients = [cv_file.getNode(node_name).mat() for node_name in nodes_list]
    cv_file.release()
    return coefficients
