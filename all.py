# for i, contour in enumerate(contours):
#     area = cv.contourArea(contour)
#     if area > 100:  # Adjust this threshold as needed
#         cv.drawContours(img, [contour], -1, (0, 255, 0), 2)
        
#         # Calculate the centroid of the contour
#         M = cv.moments(contour)
#         if M["m00"] != 0:
#             cX = int(M["m10"] / M["m00"])
#             cY = int(M["m01"] / M["m00"])
#             # Mark the center of the light spot
#             cv.circle(img, (cX, cY), 5, (255, 0, 0), -1)
#             # Print the location of the light spot
            
#             print(f"Light Spot {i+1} Location: ({cX}, {cY}) pixels")
        
#         cv.putText(img, f"Spot {i+1}", (contour[0][0][0], contour[0][0][1]),
#                     cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)