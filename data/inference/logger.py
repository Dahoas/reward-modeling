import json

class Logger:
	name: str

	@classmethod
	def init(cls, name):
		cls.name = name
		print(f"Logging in {cls.name}")

	@classmethod
	def log(cls, dicts):
		with open(f'{cls.name}.jsonl', 'a+') as f:
			for dict_t in dicts:
				json.dump(dict_t, f)
				f.write('\n')
