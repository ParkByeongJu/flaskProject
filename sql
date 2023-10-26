CREATE TABLE detected_faces (
    face_id NUMBER,
    gender VARCHAR2(10),
    age VARCHAR2(10),
    detect_time TIMESTAMP
);

select * from detected_faces;

drop table detected_faces;