import cv2
import face_recognition
 
ronaldo = face_recognition.load_image_file('ImageStore/Ronaldo.jpg')
ronaldo = cv2.cvtColor(ronaldo,cv2.COLOR_BGR2RGB)
ronaldinho = face_recognition.load_image_file('ImageStore/Ronaldinho.jpg')
ronaldinho = cv2.cvtColor(ronaldinho,cv2.COLOR_BGR2RGB)
ronaldoTest = face_recognition.load_image_file('ImageStore/Ronaldinho.jpg')
ronaldoTest = cv2.cvtColor(ronaldoTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(ronaldo)[0]
encodeRonaldo1 = face_recognition.face_encodings(ronaldo)[0]
cv2.rectangle(ronaldo,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
 
faceLocTest = face_recognition.face_locations(ronaldinho)[0]
encodeTest = face_recognition.face_encodings(ronaldinho)[0]
cv2.rectangle(ronaldinho,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
 
faceLocTest2 = face_recognition.face_locations(ronaldoTest)[0]
encodeRonaldo2 = face_recognition.face_encodings(ronaldoTest)[0]
cv2.rectangle(ronaldoTest,(faceLocTest2[3],faceLocTest2[0]),(faceLocTest2[1],faceLocTest2[2]),(255,0,255),2)

print(encodeRonaldo1)
results = face_recognition.compare_faces([encodeRonaldo1],encodeRonaldo2)
faceDis = face_recognition.face_distance([encodeRonaldo1], encodeRonaldo2)
print(results, faceDis)
# results = face_recognition.compare_faces([encodeElon],encodeTest)
# faceDis = face_recognition.face_distance([encodeElon],encodeTest)
# print(results,faceDis)
# cv2.putText(ronaldinho,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
 
cv2.imshow('ronaldo ',ronaldo)
cv2.imshow('ronaldinho',ronaldinho)
cv2.waitKey(0)
