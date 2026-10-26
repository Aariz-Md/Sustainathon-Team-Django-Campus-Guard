# Sustainathon-Team-Django-Campus-Guard

Video analytics and campus security monitoring platform ("Mini Pilot in a Box") integrating multi-camera computer vision detection modules.

## Modules

1. **aariz_work**: Entry/Exit Detection module (person tracking and boundary line-crossing detection via YOLOv8)
2. **jhithesh_work**: Zone Intrusion Detection module (polygon zone dwell and after-hours presence detection)
3. **gautham_work**: Tool/Weapon Detection module (tool slot presence, removal, and occlusion detection)
4. **sanjaay_work**: Camera Tamper & Pipeline Integration module (camera blur, cover, deflection, offline detection, and event validation)

## Setup & Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running Tests

Run the test suite across modules:

```bash
pytest
```
