import datetime

from pydantic import BaseModel


class SurveyResponseIn(BaseModel):
    id: int
    surveyId: int
    areaId:int
    userId: int
    mark: int
    comment: str
